// Function: sub_2F3BD60
// Address: 0x2f3bd60
//
__int64 __fastcall sub_2F3BD60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_97FFF0((__int64)rwlock);
  sub_2FEF6D0(rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Pre-ISel Intrinsic Lowering";
    *(_QWORD *)(v1 + 16) = "pre-isel-intrinsic-lowering";
    *(_QWORD *)(v1 + 32) = &unk_502334C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 27;
    *(_QWORD *)(v1 + 48) = sub_2F3C3E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
