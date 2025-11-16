// Function: sub_2D17840
// Address: 0x2d17840
//
__int64 __fastcall sub_2D17840(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 47;
    *(_QWORD *)v1 = "Optimize selected kernels and their call chains";
    *(_QWORD *)(v1 + 16) = "selectkernels";
    *(_QWORD *)(v1 + 32) = &unk_5015F0C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2D17EA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
