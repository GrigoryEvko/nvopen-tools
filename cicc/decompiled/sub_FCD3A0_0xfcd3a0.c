// Function: sub_FCD3A0
// Address: 0xfcd3a0
//
__int64 __fastcall sub_FCD3A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Register pressure analysis";
    *(_QWORD *)(v1 + 16) = "rpa";
    *(_QWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = &unk_4F8D474;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_FCE140;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
