// Function: sub_11FC810
// Address: 0x11fc810
//
__int64 __fastcall sub_11FC810(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 result; // rax

  v3 = *(__int64 **)(a1 + 8);
  v4 = *(__int64 **)a1;
  if ( v3 != *(__int64 **)a1 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 176);
        if ( v6 != v5 + 192 )
          _libc_free(v6, a2);
        v7 = *(_QWORD *)(v5 + 88);
        if ( v7 != v5 + 104 )
          _libc_free(v7, a2);
        sub_C7D6A0(*(_QWORD *)(v5 + 64), 8LL * *(unsigned int *)(v5 + 80), 8);
        v8 = *(_QWORD *)(v5 + 40);
        sub_BF0670(*(__int64 **)(v5 + 32), v8);
        v9 = *(_QWORD *)(v5 + 32);
        if ( v9 )
        {
          v8 = *(_QWORD *)(v5 + 48) - v9;
          j_j___libc_free_0(v9, v8);
        }
        v10 = *(_QWORD *)(v5 + 8);
        if ( v10 != v5 + 24 )
          _libc_free(v10, v8);
        a2 = 224;
        result = j_j___libc_free_0(v5, 224);
      }
      ++v4;
    }
    while ( v3 != v4 );
    v4 = *(__int64 **)a1;
  }
  if ( v4 )
    return j_j___libc_free_0(v4, *(_QWORD *)(a1 + 16) - (_QWORD)v4);
  return result;
}
