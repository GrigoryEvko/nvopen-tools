// Function: sub_30E2210
// Address: 0x30e2210
//
void __fastcall sub_30E2210(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi

  *a1 = &unk_4A326B0;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[74];
  if ( v2 )
    v2(a1 + 72, a1 + 72, 3);
  v3 = (_QWORD *)a1[22];
  if ( v3 != a1 + 24 )
    _libc_free((unsigned __int64)v3);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v4 = (_QWORD *)a1[9];
  if ( v4 != a1 + 11 )
    _libc_free((unsigned __int64)v4);
}
