// Function: sub_103A9C0
// Address: 0x103a9c0
//
__int64 __fastcall sub_103A9C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 10;
    *(_QWORD *)v1 = "Memory SSA";
    *(_QWORD *)(v1 + 16) = "memoryssa";
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 32) = &unk_4F8F808;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_103DD90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
