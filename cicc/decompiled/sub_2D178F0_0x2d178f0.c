// Function: sub_2D178F0
// Address: 0x2d178f0
//
void __fastcall sub_2D178F0(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rdi

  *a1 = &unk_4A25DD0;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[30];
  if ( v2 )
    v2(a1 + 28, a1 + 28, 3);
  v3 = a1[24];
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = a1[20];
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = a1[17];
  if ( v5 )
    j_j___libc_free_0(v5);
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13]);
  v6 = (_QWORD *)a1[9];
  if ( v6 != a1 + 11 )
    _libc_free((unsigned __int64)v6);
}
