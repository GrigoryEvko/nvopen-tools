// Function: sub_C14210
// Address: 0xc14210
//
__int64 __fastcall sub_C14210(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // r12
  _QWORD *v11; // rdi

  v3 = *(_QWORD *)(a1 + 360);
  v4 = v3 + 32LL * *(unsigned int *)(a1 + 368);
  *(_QWORD *)a1 = &unk_49DB460;
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 - 24);
      v4 -= 32;
      if ( v5 )
      {
        a2 = *(_QWORD *)(v4 + 24) - v5;
        j_j___libc_free_0(v5, a2);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 360);
  }
  if ( v4 != a1 + 376 )
    _libc_free(v4, a2);
  v6 = 16LL * *(unsigned int *)(a1 + 352);
  sub_C7D6A0(*(_QWORD *)(a1 + 336), v6, 8);
  if ( *(_DWORD *)(a1 + 316) )
  {
    v7 = *(unsigned int *)(a1 + 312);
    v8 = *(_QWORD *)(a1 + 304);
    if ( (_DWORD)v7 )
    {
      v9 = 8 * v7;
      v10 = 0;
      do
      {
        v11 = *(_QWORD **)(v8 + v10);
        if ( v11 != (_QWORD *)-8LL && v11 )
        {
          v6 = *v11 + 17LL;
          sub_C7D6A0(v11, v6, 8);
          v8 = *(_QWORD *)(a1 + 304);
        }
        v10 += 8;
      }
      while ( v9 != v10 );
    }
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 304);
  }
  _libc_free(v8, v6);
  return sub_E98B30(a1);
}
