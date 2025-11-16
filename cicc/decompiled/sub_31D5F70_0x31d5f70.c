// Function: sub_31D5F70
// Address: 0x31d5f70
//
void __fastcall sub_31D5F70(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v3; // rdi
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rdi

  *a1 = &unk_4A34FB8;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[75];
  if ( v2 )
    v2(a1 + 73, a1 + 73, 3);
  v3 = (_QWORD *)a1[23];
  if ( v3 != a1 + 25 )
    _libc_free((unsigned __int64)v3);
  v4 = a1[18];
  if ( v4 )
    j_j___libc_free_0(v4);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v5 = (_QWORD *)a1[9];
  if ( v5 != a1 + 11 )
    _libc_free((unsigned __int64)v5);
}
