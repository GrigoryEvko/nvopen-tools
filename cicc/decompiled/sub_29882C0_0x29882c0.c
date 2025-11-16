// Function: sub_29882C0
// Address: 0x29882c0
//
__int64 __fastcall sub_29882C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_10564E0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_22DC340((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Structurize the CFG";
    *(_QWORD *)(v1 + 16) = "structurizecfg";
    *(_QWORD *)(v1 + 32) = &unk_500778C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_298B0B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
