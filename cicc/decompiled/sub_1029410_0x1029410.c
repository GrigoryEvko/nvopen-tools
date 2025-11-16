// Function: sub_1029410
// Address: 0x1029410
//
__int64 __fastcall sub_1029410(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Memory Dependence Analysis";
    *(_QWORD *)(v1 + 16) = "memdep";
    *(_QWORD *)(v1 + 24) = 6;
    *(_QWORD *)(v1 + 32) = &unk_4F8EE5C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_102D110;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
