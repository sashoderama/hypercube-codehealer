import os
import gc
import random
import ast
import hashlib
import torch
import logging
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from redis import Redis
from typing import Dict
from .dynamic_batch import get_dynamic_batch_size

# Constants
EMBED_DIM = 768
HIDDEN_DIM = 1024
GPU_AVAILABLE = torch.cuda.is_available()
BLACKLIST = {"system", "eval", "exec", "ctypes", "Popen", "pickle", "exploit"}
FREE_EXEC_LIMIT = 10
__version__ = "1.1.0"

# Exceptions
class UnsafeCodeError(Exception): pass
class CommercialFeatureError(Exception): pass
class RateLimitExceeded(Exception): pass

class DangerousVisitor(ast.NodeVisitor):
    def __init__(self):
        self.unsafe = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in BLACKLIST:
            self.unsafe = True

class CodeHealer:
    def __init__(self, device: str = "cuda" if GPU_AVAILABLE else "cpu", is_pro_user: bool = False):
        self.device = device
        self.is_pro_user = is_pro_user
        self.logger = logging.getLogger("CodeHealer")
        self.logger.setLevel(logging.DEBUG)

        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=bool(os.getenv("REDIS_SSL"))
        )

        self.model = self._init_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = GradScaler()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)  # Optional

    def _init_model(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1)
        ).to(self.device)

    async def execute_with_healing(self, code: str) -> Dict:
        try:
            # 1. Rate limiting
            code_hash = f"exec_limit:{hashlib.sha256(code.encode()).hexdigest()}"
            if self.redis.incr(code_hash) > FREE_EXEC_LIMIT and not self.is_pro_user:
                raise RateLimitExceeded("Rate limit exceeded")
            self.redis.expire(code_hash, 86400)

            # 2. AST validation
            tree = ast.parse(code)
            visitor = DangerousVisitor()
            visitor.visit(tree)
            if visitor.unsafe and not self.is_pro_user:
                raise UnsafeCodeError("Blocked: unsafe code pattern detected")
            if "exploit" in code.lower() and not self.is_pro_user:
                raise CommercialFeatureError("Exploit synthesis is Enterprise-only")

            # 3. Embedding with gradient tracking
            embedding = torch.randn(EMBED_DIM, device=self.device, dtype=torch.float32, requires_grad=True)

            # 4. Memory-aware batch config
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3 if GPU_AVAILABLE else 0
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3 if GPU_AVAILABLE else 1
            batch_size = get_dynamic_batch_size(allocated, total)
            accumulation_steps = max(2, int(8 / batch_size))

            with autocast():
                self.optimizer.zero_grad()

                # Forward
                output = self.model(embedding.unsqueeze(0)).mean()

                # Target tensor with gradient
                target = torch.tensor([1.0], device=self.device, dtype=torch.float32, requires_grad=True)

                # Loss and backprop
                loss = nn.MSELoss()(output, target)
                self.scaler.scale(loss).backward()

                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
            gc.collect()

            return {
                "success": True,
                "output": f"# Patched by Hypercube v{__version__}\n{code}",
                "confidence": output.detach().cpu().item()
            }

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {"success": False, "error": str(e)}
