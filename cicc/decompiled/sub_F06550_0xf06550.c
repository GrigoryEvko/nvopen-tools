// Function: sub_F06550
// Address: 0xf06550
//
__int64 __fastcall sub_F06550(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r13

  sub_CFB980((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_D1D8C0((__int64)rwlock);
  sub_1049990(rwlock);
  sub_1027850(rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Combine redundant instructions";
    *(_QWORD *)(v1 + 16) = "instcombine";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_4F8AED0;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_F11220;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
