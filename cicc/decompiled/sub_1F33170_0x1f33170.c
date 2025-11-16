// Function: sub_1F33170
// Address: 0x1f33170
//
__int64 __fastcall sub_1F33170(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 v7; // rdi

  v2 = *(unsigned int *)(a1 + 392);
  *(_QWORD *)a1 = off_49FED80;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 376);
    v4 = v3 + 32 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 8);
          if ( v5 )
            break;
        }
        v3 += 32;
        if ( v4 == v3 )
          goto LABEL_7;
      }
      v6 = *(_QWORD *)(v3 + 24);
      v3 += 32;
      j_j___libc_free_0(v5, v6 - v5);
    }
    while ( v4 != v3 );
  }
LABEL_7:
  j___libc_free_0(*(_QWORD *)(a1 + 376));
  v7 = *(_QWORD *)(a1 + 288);
  if ( v7 != a1 + 304 )
    _libc_free(v7);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 408);
}
