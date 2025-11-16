// Function: sub_2F3C4C0
// Address: 0x2f3c4c0
//
void __fastcall sub_2F3C4C0(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v3; // rdi

  *a1 = &unk_4A2AA68;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[24];
  if ( v2 )
    v2(a1 + 22, a1 + 22, 3);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v3 = (_QWORD *)a1[9];
  if ( v3 != a1 + 11 )
    _libc_free((unsigned __int64)v3);
}
