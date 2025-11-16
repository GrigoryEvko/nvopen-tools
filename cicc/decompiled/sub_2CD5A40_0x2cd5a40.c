// Function: sub_2CD5A40
// Address: 0x2cd5a40
//
__int64 __fastcall sub_2CD5A40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_31C5560(rwlock);
  sub_2C6F190((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Lower structure arguments";
    *(_QWORD *)(v1 + 16) = "lower-struct-args";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_4CE0068;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2CD6230;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
