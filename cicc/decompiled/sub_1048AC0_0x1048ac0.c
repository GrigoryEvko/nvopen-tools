// Function: sub_1048AC0
// Address: 0x1048ac0
//
__int64 __fastcall sub_1048AC0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_1027A60((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Optimization Remark Emitter";
    *(_QWORD *)(v1 + 16) = "opt-remark-emitter";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_4F8FAE4;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_1049B00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
