// Function: sub_D00190
// Address: 0xd00190
//
__int64 __fastcall sub_D00190(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "Basic Alias Analysis (stateless AA impl)";
    *(_QWORD *)(v1 + 16) = "basic-aa";
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 32) = &unk_4F8670C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_D055F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
