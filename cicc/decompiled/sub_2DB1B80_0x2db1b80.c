// Function: sub_2DB1B80
// Address: 0x2db1b80
//
__int64 __fastcall sub_2DB1B80(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030();
  sub_2E6D3E0(rwlock);
  sub_2EE7B50(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Early If Converter";
    *(_QWORD *)(v1 + 16) = "early-ifcvt";
    *(_QWORD *)(v1 + 32) = &unk_501CF6C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2DB2090;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
