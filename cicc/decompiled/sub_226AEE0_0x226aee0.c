// Function: sub_226AEE0
// Address: 0x226aee0
//
void __fastcall sub_226AEE0(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v3; // rdi

  *a1 = &unk_4A08488;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[21];
  if ( v2 )
    v2(a1 + 19, a1 + 19, 3);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v3 = (_QWORD *)a1[9];
  if ( v3 != a1 + 11 )
    _libc_free((unsigned __int64)v3);
}
