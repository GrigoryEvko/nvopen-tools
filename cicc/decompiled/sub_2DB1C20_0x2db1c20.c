// Function: sub_2DB1C20
// Address: 0x2db1c20
//
__int64 __fastcall sub_2DB1C20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0(rwlock);
  sub_2E44030(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Early If Predicator";
    *(_QWORD *)(v1 + 16) = "early-if-predicator";
    *(_QWORD *)(v1 + 32) = &unk_501CF64;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2DB2310;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
