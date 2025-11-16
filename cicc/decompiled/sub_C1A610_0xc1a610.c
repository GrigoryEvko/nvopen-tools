// Function: sub_C1A610
// Address: 0xc1a610
//
void __fastcall sub_C1A610(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  _QWORD *v4; // rdi

  *a1 = &unk_49DB9B8;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[24];
  if ( v3 )
  {
    a2 = (char *)(a1 + 22);
    v3(a2, a2, 3);
  }
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v4 = (_QWORD *)a1[9];
  if ( v4 != a1 + 11 )
    _libc_free(v4, a2);
}
