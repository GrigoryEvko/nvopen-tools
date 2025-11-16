// Function: sub_353D140
// Address: 0x353d140
//
__int64 __fastcall sub_353D140(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2E10620((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Modulo Software Pipelining";
    *(_QWORD *)(v1 + 16) = "pipeliner";
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 32) = &unk_503DCCC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_35425F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
