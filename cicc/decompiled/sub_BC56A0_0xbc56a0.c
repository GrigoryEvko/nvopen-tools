// Function: sub_BC56A0
// Address: 0xbc56a0
//
void __fastcall sub_BC56A0(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  _QWORD *v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  _QWORD *v8; // rdi

  *a1 = &unk_49DB1D8;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[81];
  if ( v3 )
  {
    a2 = (char *)(a1 + 79);
    v3(a2, a2, 3);
  }
  v4 = (_QWORD *)a1[29];
  if ( v4 != a1 + 31 )
    _libc_free(v4, a2);
  v5 = a1[24];
  if ( v5 )
  {
    a2 = (char *)(a1[26] - v5);
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[20];
  if ( v6 )
  {
    a2 = (char *)(a1[22] - v6);
    j_j___libc_free_0(v6, a2);
  }
  v7 = a1[17];
  if ( v7 )
  {
    a2 = (char *)(a1[19] - v7);
    j_j___libc_free_0(v7, a2);
  }
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v8 = (_QWORD *)a1[9];
  if ( v8 != a1 + 11 )
    _libc_free(v8, a2);
}
