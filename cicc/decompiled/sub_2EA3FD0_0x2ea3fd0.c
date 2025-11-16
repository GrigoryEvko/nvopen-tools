// Function: sub_2EA3FD0
// Address: 0x2ea3fd0
//
__int64 __fastcall sub_2EA3FD0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 33;
    *(_QWORD *)v1 = "Machine Natural Loop Construction";
    *(_QWORD *)(v1 + 16) = "machine-loops";
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 32) = &unk_50208AC;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2EA63D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
