// Function: sub_2D0EB50
// Address: 0x2d0eb50
//
__int64 __fastcall sub_2D0EB50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Reuses local memory if possible";
    *(_QWORD *)(v1 + 16) = "nvvm-reuse-local-memory";
    *(_QWORD *)(v1 + 24) = 23;
    *(_QWORD *)(v1 + 32) = &unk_5015E2C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D0F0D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
