// Function: sub_CA6E30
// Address: 0xca6e30
//
__int64 __fastcall sub_CA6E30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r13
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 *v9; // rdi
  __int64 *v10; // r14
  __int64 *v11; // rbx
  __int64 i; // rax
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rbx
  __int64 *v16; // r13
  __int64 v17; // rdi
  __int64 result; // rax
  __int64 v19; // rdi

  v3 = *(_QWORD *)(a1 + 224);
  if ( v3 != a1 + 240 )
    _libc_free(v3, a2);
  v4 = *(_QWORD *)(a1 + 192);
  if ( v4 != a1 + 208 )
    _libc_free(v4, a2);
  v5 = *(unsigned __int64 **)(a1 + 184);
  v6 = (unsigned __int64 *)(a1 + 176);
  while ( v6 != v5 )
  {
    while ( 1 )
    {
      v7 = v5;
      v5 = (unsigned __int64 *)v5[1];
      v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      *v5 = v8 | *v5 & 7;
      *(_QWORD *)(v8 + 8) = v5;
      v9 = (unsigned __int64 *)v7[5];
      *v7 &= 7u;
      v7[1] = 0;
      if ( v9 == v7 + 7 )
        break;
      a2 = v7[7] + 1;
      j_j___libc_free_0(v9, a2);
      if ( v6 == v5 )
        goto LABEL_9;
    }
  }
LABEL_9:
  v10 = *(__int64 **)(a1 + 96);
  v11 = &v10[*(unsigned int *)(a1 + 104)];
  if ( v10 != v11 )
  {
    for ( i = *(_QWORD *)(a1 + 96); ; i = *(_QWORD *)(a1 + 96) )
    {
      v13 = *v10;
      v14 = (unsigned int)(((__int64)v10 - i) >> 3) >> 7;
      a2 = 4096LL << v14;
      if ( v14 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v10;
      sub_C7D6A0(v13, a2, 16);
      if ( v11 == v10 )
        break;
    }
  }
  v15 = *(__int64 **)(a1 + 144);
  v16 = &v15[2 * *(unsigned int *)(a1 + 152)];
  if ( v15 != v16 )
  {
    do
    {
      a2 = v15[1];
      v17 = *v15;
      v15 += 2;
      sub_C7D6A0(v17, a2, 16);
    }
    while ( v16 != v15 );
    v16 = *(__int64 **)(a1 + 144);
  }
  result = a1 + 160;
  if ( v16 != (__int64 *)(a1 + 160) )
    result = _libc_free(v16, a2);
  v19 = *(_QWORD *)(a1 + 96);
  if ( v19 != a1 + 112 )
    return _libc_free(v19, a2);
  return result;
}
