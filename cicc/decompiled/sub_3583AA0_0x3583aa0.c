// Function: sub_3583AA0
// Address: 0x3583aa0
//
__int64 __fastcall sub_3583AA0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E399F0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2EB3F30((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  sub_2EAFCC0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Load MIR Sample Profile";
    *(_QWORD *)(v1 + 16) = "fs-profile-loader";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_503F254;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_35852A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
