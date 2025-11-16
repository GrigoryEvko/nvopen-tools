// Function: sub_163BE40
// Address: 0x163be40
//
__int64 *__fastcall sub_163BE40(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 *v5; // r15
  __int64 *v6; // rdx
  int v7; // ebx
  __int64 *result; // rax
  __int64 *v9; // r9
  int v10; // r13d
  int v11; // r8d
  __int64 v12; // r11
  __int64 *v13; // rsi
  int v14; // r10d
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 *v19; // rdi
  __int64 v20; // r8
  int v21; // edi
  int v22; // r10d
  __int64 v23; // rdi
  int v24; // [rsp+4h] [rbp-2Ch]

  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  v6 = &v5[v4];
  v7 = *(_DWORD *)(a1 + 24);
  result = v6;
  if ( *(_DWORD *)(a1 + 16) && v5 != v6 )
  {
    result = *(__int64 **)(a1 + 8);
    do
    {
      if ( *result != -8 && *result != -16 )
        break;
      ++result;
    }
    while ( result != v6 );
  }
LABEL_2:
  v9 = &v5[v4];
  v10 = v7 - 1;
  while ( v9 != result )
  {
    v11 = *(_DWORD *)(a2 + 24);
    v12 = *(_QWORD *)(a2 + 8);
    v13 = result;
    v14 = v11 - 1;
    while ( 1 )
    {
      do
        ++result;
      while ( result != v6 && (*result == -8 || *result == -16) );
      v15 = *v13;
      if ( !v11 )
        break;
      v16 = v14 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v17 = *(_QWORD *)(v12 + 8LL * v16);
      if ( v15 != v17 )
      {
        v24 = 1;
        while ( v17 != -8 )
        {
          v16 = v14 & (v24 + v16);
          ++v24;
          v17 = *(_QWORD *)(v12 + 8LL * v16);
          if ( v15 == v17 )
            goto LABEL_10;
        }
        break;
      }
LABEL_10:
      if ( v9 == result )
        return result;
      v13 = result;
    }
    if ( v7 )
    {
      v18 = v10 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v19 = &v5[v18];
      v20 = *v19;
      if ( v15 == *v19 )
      {
LABEL_16:
        *v19 = -16;
        v4 = *(unsigned int *)(a1 + 24);
        --*(_DWORD *)(a1 + 16);
        v5 = *(__int64 **)(a1 + 8);
        ++*(_DWORD *)(a1 + 20);
        v7 = v4;
        goto LABEL_2;
      }
      v21 = 1;
      while ( v20 != -8 )
      {
        v22 = v21 + 1;
        v23 = v10 & (v18 + v21);
        v18 = v23;
        v19 = &v5[v23];
        v20 = *v19;
        if ( v15 == *v19 )
          goto LABEL_16;
        v21 = v22;
      }
    }
  }
  return result;
}
