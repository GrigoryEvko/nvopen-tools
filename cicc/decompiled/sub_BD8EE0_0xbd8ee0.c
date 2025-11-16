// Function: sub_BD8EE0
// Address: 0xbd8ee0
//
__int64 __fastcall sub_BD8EE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 15;
    *(_QWORD *)v1 = "Module Verifier";
    *(_QWORD *)(v1 + 16) = "verify";
    *(_QWORD *)(v1 + 32) = &unk_4F836D4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 6;
    *(_QWORD *)(v1 + 48) = sub_BE0850;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
