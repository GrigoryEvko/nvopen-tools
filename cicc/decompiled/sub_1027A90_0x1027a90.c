// Function: sub_1027A90
// Address: 0x1027a90
//
__int64 __fastcall sub_1027A90(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Lazy Branch Probability Analysis";
    *(_QWORD *)(v1 + 16) = "lazy-branch-prob";
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 32) = &unk_4F8EE50;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_1028470;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
