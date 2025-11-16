// Function: sub_24FB1A0
// Address: 0x24fb1a0
//
__int64 __fastcall sub_24FB1A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Inliner for always_inline functions";
    *(_QWORD *)(v1 + 16) = "always-inline";
    *(_QWORD *)(v1 + 32) = &unk_4FEE4AC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_24FB4E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
