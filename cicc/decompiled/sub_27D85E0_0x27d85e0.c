// Function: sub_27D85E0
// Address: 0x27d85e0
//
__int64 __fastcall sub_27D85E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Remove redundant instructions";
    *(_QWORD *)(v1 + 16) = "instsimplify";
    *(_QWORD *)(v1 + 32) = &unk_4FFD72C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_27D87C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
