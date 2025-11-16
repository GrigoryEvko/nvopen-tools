// Function: sub_34E6310
// Address: 0x34e6310
//
__int64 __fastcall sub_34E6310(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 12;
    *(_QWORD *)v1 = "If Converter";
    *(_QWORD *)(v1 + 16) = "if-converter";
    *(_QWORD *)(v1 + 32) = &unk_503B14C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_34EAFC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
