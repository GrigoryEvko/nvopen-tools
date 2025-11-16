// Function: sub_34E6D50
// Address: 0x34e6d50
//
void __fastcall sub_34E6D50(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *a1 = off_4A384A8;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[81];
  if ( v2 )
    v2(a1 + 79, a1 + 79, 3);
  v3 = a1[76];
  if ( v3 )
    _libc_free(v3);
  v4 = a1[71];
  if ( (_QWORD *)v4 != a1 + 74 )
    _libc_free(v4);
  v5 = a1[54];
  if ( (_QWORD *)v5 != a1 + 56 )
    _libc_free(v5);
  v6 = a1[26];
  v7 = a1[25];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 216);
      if ( v8 != v7 + 232 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v7 + 40);
      if ( v9 != v7 + 56 )
        _libc_free(v9);
      v7 += 392LL;
    }
    while ( v6 != v7 );
    v7 = a1[25];
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
