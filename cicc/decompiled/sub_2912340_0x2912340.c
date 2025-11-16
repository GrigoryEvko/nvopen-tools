// Function: sub_2912340
// Address: 0x2912340
//
__int64 __fastcall sub_2912340(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Scalar Replacement Of Aggregates";
    *(_QWORD *)(v1 + 16) = "sroa";
    *(_QWORD *)(v1 + 32) = &unk_5005390;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 48) = sub_291E8D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
