// Function: sub_13C92C0
// Address: 0x13c92c0
//
__int64 __fastcall sub_13C92C0(_QWORD *a1)
{
  _QWORD *v2; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 *v4; // r15
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  __int64 v9; // rcx
  unsigned __int64 v10; // rdi

  v2 = (_QWORD *)a1[20];
  *a1 = &unk_49EA558;
  if ( v2 )
  {
    v3 = v2[30];
    if ( v3 != v2[29] )
      _libc_free(v3);
    v4 = (unsigned __int64 *)v2[27];
    while ( v2 + 26 != v4 )
    {
      v5 = v4;
      v4 = (unsigned __int64 *)v4[1];
      v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
      *v4 = v6 | *v4 & 7;
      *(_QWORD *)(v6 + 8) = v4;
      v7 = v5[8];
      *v5 &= 7u;
      v5[1] = 0;
      *(v5 - 4) = (unsigned __int64)&unk_49EA628;
      if ( v7 != v5[7] )
        _libc_free(v7);
      v8 = v5[5];
      if ( v8 != 0 && v8 != -8 && v8 != -16 )
        sub_1649B30(v5 + 3);
      *(v5 - 4) = (unsigned __int64)&unk_49EE2B0;
      v9 = *(v5 - 1);
      if ( v9 != -8 && v9 != 0 && v9 != -16 )
        sub_1649B30(v5 - 3);
      j_j___libc_free_0(v5 - 4, 136);
    }
    v10 = v2[7];
    if ( v10 != v2[6] )
      _libc_free(v10);
    j_j___libc_free_0(v2, 520);
  }
  *a1 = &unk_49EAEF0;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
