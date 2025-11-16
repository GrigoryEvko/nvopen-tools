// Function: sub_2D56770
// Address: 0x2d56770
//
__int64 __fastcall sub_2D56770(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2D50D40((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_2FEF6D0(rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 28;
    *(_QWORD *)v1 = "Optimize for code generation";
    *(_QWORD *)(v1 + 16) = "codegenprepare";
    *(_QWORD *)(v1 + 32) = &unk_501696C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_2D5CC90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
