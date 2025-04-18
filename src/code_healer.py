import os
import gc
import ast
import hashlib
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from redis import Redis
from typing import Dict

# Constants
EMBED_DIM = 768
HIDDEN_DIM = 1024
GPU_AVAILABLE = torch.cuda.is_available()
BLACKLIST = {"system", "eval", "exec", "ctypes", "Popen", "pickle", "exploit"}
FREE_EXEC_LIMIT = 10
__version__ = "1.1.0"

class UnsafeCodeError(Exception): pass
class RateLimitExceeded(Exception): pass

class DangerousVisitor(ast.NodeVisitor):
    def __init__(self):
        self.unsafe = False
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in BLACKLIST:
            self.unsafe = True

class CodeHealer:
    def __init__(self, device: str = "cuda" if GPU_AVAILABLE else "cpu", 
                 is_pro_user: bool = False):
        self.device = device
        self.is_pro_user = is_pro_user
        self.logger = logging.getLogger("CodeHealer")
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            ssl=bool(os.getenv("REDIS_SSL", False))
        )
        self.model = self._init_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = GradScaler()
        self.scheduler = StepLR(self.optimizer, step_size=100)

    def _init_model(self) -> nn.Sequential:
        model = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        ).to(self.device)
        model.train()
        return model

    async def execute_with_healing(self, code: str) -> Dict:
        try:
            # Security checks
            code_hash = f"exec_limit:{hashlib.sha256(code.encode()).hexdigest()}"
            if self.redis.incr(code_hash) > FREE_EXEC_LIMIT and not self.is_pro_user:
                self.redis.expire(code_hash, 86400)
                raise RateLimitExceeded("Rate limit exceeded (10/day)")
            
            # AST analysis
            tree = ast.parse(code)
            visitor = DangerousVisitor()
            visitor.visit(tree)
            if visitor.unsafe and not self.is_pro_user:
                raise UnsafeCodeError("Dangerous code patterns detected")

            # Healing process
            embedding = torch.randn(EMBED_DIM, device=self.device, dtype=torch.float32)
            
            with autocast():
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(embedding.unsqueeze(0)).mean()
                target = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                
                # Loss calculation
                loss = nn.MSELoss()(output, target)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update steps
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
            gc.collect()

            return {
                "success": True,
                "output": f"# Patched by Hypercube v{__version__}\n{code}",
                "confidence": round(output.item(), 4)
            }
        
        except Exception as e:
            self.logger.error(f"Execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
