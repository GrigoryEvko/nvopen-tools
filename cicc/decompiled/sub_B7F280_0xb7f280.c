// Function: sub_B7F280
// Address: 0xb7f280
//
void __fastcall sub_B7F280(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi

  *a1 = &off_49DA648;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[74];
  if ( v3 )
  {
    a2 = (char *)(a1 + 72);
    v3(a2, a2, 3);
  }
  v4 = (_QWORD *)a1[22];
  if ( v4 != a1 + 24 )
    _libc_free(v4, a2);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v5 = (_QWORD *)a1[9];
  if ( v5 != a1 + 11 )
    _libc_free(v5, a2);
}
