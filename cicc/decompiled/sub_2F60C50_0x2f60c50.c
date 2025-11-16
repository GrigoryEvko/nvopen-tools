// Function: sub_2F60C50
// Address: 0x2f60c50
//
__int64 __fastcall sub_2F60C50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E10620((__int64)rwlock);
  sub_2FACF50(rwlock);
  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Register Coalescer";
    *(_QWORD *)(v1 + 16) = "register-coalescer";
    *(_QWORD *)(v1 + 32) = &unk_502476C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2F65F20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
