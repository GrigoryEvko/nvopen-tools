// Function: sub_B7F3A0
// Address: 0xb7f3a0
//
void __fastcall sub_B7F3A0(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  _QWORD *v7; // rdi

  *a1 = &unk_49DA6C8;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[30];
  if ( v3 )
  {
    a2 = (char *)(a1 + 28);
    v3(a2, a2, 3);
  }
  v4 = a1[24];
  if ( v4 )
  {
    a2 = (char *)(a1[26] - v4);
    j_j___libc_free_0(v4, a2);
  }
  v5 = a1[20];
  if ( v5 )
  {
    a2 = (char *)(a1[22] - v5);
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[17];
  if ( v6 )
  {
    a2 = (char *)(a1[19] - v6);
    j_j___libc_free_0(v6, a2);
  }
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v7 = (_QWORD *)a1[9];
  if ( v7 != a1 + 11 )
    _libc_free(v7, a2);
}
