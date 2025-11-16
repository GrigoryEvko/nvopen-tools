// Function: sub_1D5B410
// Address: 0x1d5b410
//
_QWORD *__fastcall sub_1D5B410(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // r9
  unsigned int *v4; // rdx
  unsigned int *i; // r8
  __int64 v6; // r8
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v24; // rbx
  _QWORD *result; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  _QWORD *v28; // rdx
  __int64 v29; // rsi

  v2 = *(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( v2[5] )
      sub_15F2070(*(_QWORD **)(a1 + 8));
    sub_15F2180((__int64)v2, *(_QWORD *)(a1 + 16));
  }
  else
  {
    v29 = sub_157EE30(*(_QWORD *)(a1 + 16));
    if ( v29 )
      v29 -= 24;
    if ( v2[5] )
      sub_15F22F0(v2, v29);
    else
      sub_15F2120((__int64)v2, v29);
  }
  v3 = *(_QWORD *)(a1 + 96);
  if ( v3 )
  {
    v4 = *(unsigned int **)(v3 + 16);
    for ( i = &v4[4 * *(unsigned int *)(v3 + 24)]; i != v4; v4 += 4 )
    {
      v21 = *(_QWORD *)v4;
      v22 = *(_QWORD *)(v3 + 8);
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 23LL) & 0x40) != 0 )
        v16 = *(_QWORD *)(v21 - 8);
      else
        v16 = v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
      v17 = (_QWORD *)(v16 + 24LL * v4[2]);
      if ( *v17 )
      {
        v18 = v17[1];
        v19 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *v17 = v22;
      if ( v22 )
      {
        v20 = *(_QWORD *)(v22 + 8);
        v17[1] = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (unsigned __int64)(v17 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
        v17[2] = (v22 + 8) | v17[2] & 3LL;
        *(_QWORD *)(v22 + 8) = v17;
      }
    }
  }
  v6 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v6 )
  {
    v7 = 8 * v6;
    v8 = 0;
    do
    {
      v14 = *(_QWORD *)(a1 + 40);
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + v8);
      if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
        v9 = *(_QWORD *)(v14 - 8);
      else
        v9 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
      v10 = (_QWORD *)(3 * v8 + v9);
      if ( *v10 )
      {
        v11 = v10[1];
        v12 = v10[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v12 = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
      }
      *v10 = v15;
      if ( v15 )
      {
        v13 = *(_QWORD *)(v15 + 8);
        v10[1] = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = (unsigned __int64)(v10 + 1) | *(_QWORD *)(v13 + 16) & 3LL;
        v10[2] = (v15 + 8) | v10[2] & 3LL;
        *(_QWORD *)(v15 + 8) = v10;
      }
      v8 += 8;
    }
    while ( v7 != v8 );
  }
  v23 = *(_QWORD *)(a1 + 104);
  v24 = *(_QWORD *)(a1 + 8);
  result = *(_QWORD **)(v23 + 8);
  if ( *(_QWORD **)(v23 + 16) == result )
  {
    v28 = &result[*(unsigned int *)(v23 + 28)];
    if ( result == v28 )
    {
LABEL_43:
      result = v28;
    }
    else
    {
      while ( v24 != *result )
      {
        if ( v28 == ++result )
          goto LABEL_43;
      }
    }
  }
  else
  {
    result = sub_16CC9F0(v23, v24);
    if ( v24 == *result )
    {
      v26 = *(_QWORD *)(v23 + 16);
      if ( v26 == *(_QWORD *)(v23 + 8) )
        v27 = *(unsigned int *)(v23 + 28);
      else
        v27 = *(unsigned int *)(v23 + 24);
      v28 = (_QWORD *)(v26 + 8 * v27);
    }
    else
    {
      result = *(_QWORD **)(v23 + 16);
      if ( result != *(_QWORD **)(v23 + 8) )
        return result;
      result += *(unsigned int *)(v23 + 28);
      v28 = result;
    }
  }
  if ( v28 != result )
  {
    *result = -2;
    ++*(_DWORD *)(v23 + 32);
  }
  return result;
}
