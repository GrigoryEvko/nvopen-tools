// Function: sub_35AB9A0
// Address: 0x35ab9a0
//
__int64 __fastcall sub_35AB9A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2EAFCC0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 48;
    *(_QWORD *)v1 = "Prologue/Epilogue Insertion & Frame Finalization";
    *(_QWORD *)(v1 + 16) = "prologepilog";
    *(_QWORD *)(v1 + 32) = &unk_503FCFC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_35ADE90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
