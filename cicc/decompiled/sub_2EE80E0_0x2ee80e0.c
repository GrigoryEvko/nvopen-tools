// Function: sub_2EE80E0
// Address: 0x2ee80e0
//
void __fastcall sub_2EE80E0(_QWORD *a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = a1 + 50;
  sub_2EE8070((__int64)a1);
  v3 = a1[51];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 24LL))(v3);
  v4 = a1[50];
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL))(v4);
  v5 = a1[48];
  if ( v2 != (_QWORD *)v5 )
    _libc_free(v5);
  v6 = a1[42];
  if ( (_QWORD *)v6 != a1 + 44 )
    _libc_free(v6);
  v7 = a1[31];
  if ( (_QWORD *)v7 != a1 + 33 )
    _libc_free(v7);
}
