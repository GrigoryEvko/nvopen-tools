// Function: sub_31C5470
// Address: 0x31c5470
//
__int64 __fastcall sub_31C5470(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 45;
    *(_QWORD *)v1 = "Container for architecture-dependent features";
    *(_QWORD *)(v1 + 16) = "opt-arch-features";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_5035D54;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_31C5820;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
