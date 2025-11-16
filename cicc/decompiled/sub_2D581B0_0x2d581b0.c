// Function: sub_2D581B0
// Address: 0x2d581b0
//
__int64 **__fastcall sub_2D581B0(__int64 a1)
{
  unsigned int *v2; // rcx
  unsigned int *i; // r8
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 v12; // rdi
  __int64 *v13; // r12
  __int64 **result; // rax
  __int64 *j; // r13
  __int64 v16; // rdi

  v2 = *(unsigned int **)(a1 + 16);
  for ( i = &v2[4 * *(unsigned int *)(a1 + 24)]; i != v2; v2 += 4 )
  {
    v8 = *(_QWORD *)v2;
    v9 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(*(_QWORD *)v2 + 7LL) & 0x40) != 0 )
      v4 = *(_QWORD *)(v8 - 8);
    else
      v4 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
    v5 = 32LL * v2[2] + v4;
    if ( *(_QWORD *)v5 )
    {
      v6 = *(_QWORD *)(v5 + 8);
      **(_QWORD **)(v5 + 16) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 + 16);
    }
    *(_QWORD *)v5 = v9;
    if ( v9 )
    {
      v7 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v5 + 8) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = v5 + 8;
      *(_QWORD *)(v5 + 16) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v5;
    }
  }
  v10 = *(__int64 **)(a1 + 96);
  v11 = &v10[*(unsigned int *)(a1 + 104)];
  while ( v11 != v10 )
  {
    v12 = *v10++;
    sub_B59720(v12, *(_QWORD *)(a1 + 144), *(unsigned __int8 **)(a1 + 8));
  }
  v13 = *(__int64 **)(a1 + 120);
  result = (__int64 **)*(unsigned int *)(a1 + 128);
  for ( j = &v13[(_QWORD)result];
        j != v13;
        result = sub_B13360(v16, *(unsigned __int8 **)(a1 + 144), *(unsigned __int8 **)(a1 + 8), 0) )
  {
    v16 = *v13++;
  }
  return result;
}
