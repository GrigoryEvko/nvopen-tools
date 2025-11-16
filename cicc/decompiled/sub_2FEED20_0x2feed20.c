// Function: sub_2FEED20
// Address: 0x2feed20
//
void __fastcall sub_2FEED20(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi

  *a1 = &unk_4A2D8E8;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[84];
  if ( v2 )
    v2(a1 + 82, a1 + 82, 3);
  v3 = (_QWORD *)a1[24];
  qword_5023870 = 0;
  if ( v3 != a1 + 26 )
    _libc_free((unsigned __int64)v3);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v4 = (_QWORD *)a1[9];
  if ( v4 != a1 + 11 )
    _libc_free((unsigned __int64)v4);
}
