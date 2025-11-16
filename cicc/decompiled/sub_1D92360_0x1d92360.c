// Function: sub_1D92360
// Address: 0x1d92360
//
__int64 __fastcall sub_1D92360(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  *a1 = off_49FA898;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[84];
  if ( v2 )
    v2(a1 + 82, a1 + 82, 3);
  _libc_free(a1[79]);
  v3 = a1[73];
  if ( (_QWORD *)v3 != a1 + 75 )
    _libc_free(v3);
  v4 = a1[56];
  if ( (_QWORD *)v4 != a1 + 58 )
    _libc_free(v4);
  v5 = a1[30];
  v6 = a1[29];
  if ( v5 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 + 216);
      if ( v7 != v6 + 232 )
        _libc_free(v7);
      v8 = *(_QWORD *)(v6 + 40);
      if ( v8 != v6 + 56 )
        _libc_free(v8);
      v6 += 392;
    }
    while ( v5 != v6 );
    v6 = a1[29];
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[31] - v6);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 688);
}
