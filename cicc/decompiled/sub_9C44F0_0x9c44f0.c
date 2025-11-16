// Function: sub_9C44F0
// Address: 0x9c44f0
//
void __fastcall sub_9C44F0(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  _QWORD *v4; // rdi

  *a1 = &unk_49D97F0;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[23];
  if ( v3 )
  {
    a2 = (char *)(a1 + 21);
    v3(a2, a2, 3);
  }
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v4 = (_QWORD *)a1[9];
  if ( v4 != a1 + 11 )
    _libc_free(v4, a2);
}
