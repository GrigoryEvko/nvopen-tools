// Function: sub_1400330
// Address: 0x1400330
//
__int64 *__fastcall sub_1400330(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r8d
  __int64 v6; // rdi
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 *result; // rax
  __int64 *v12; // r8
  __int64 *v13; // rdi
  unsigned int v14; // r8d
  __int64 *v15; // rcx
  int v16; // r11d
  __int64 *v17; // r10
  int v18; // edi
  int v19; // edx
  int v20; // esi
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v5 = *(_DWORD *)(a3 + 24);
  v21 = a2;
  if ( !v5 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_34;
  }
  v6 = *(_QWORD *)(a3 + 8);
  v7 = a2;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v7 != *v9 )
  {
    v16 = 1;
    v17 = 0;
    while ( v10 != -8 )
    {
      if ( !v17 && v10 == -16 )
        v17 = v9;
      v8 = (v5 - 1) & (v16 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v7 == *v9 )
        goto LABEL_3;
      ++v16;
    }
    v18 = *(_DWORD *)(a3 + 16);
    if ( v17 )
      v9 = v17;
    ++*(_QWORD *)a3;
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a3 + 20) - v19 > v5 >> 3 )
      {
LABEL_30:
        *(_DWORD *)(a3 + 16) = v19;
        if ( *v9 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v9 = v7;
        v10 = v21;
        v9[1] = 0;
        goto LABEL_3;
      }
      v20 = v5;
LABEL_35:
      sub_1400170(a3, v20);
      sub_13FD8B0(a3, &v21, v22);
      v9 = (__int64 *)v22[0];
      v7 = v21;
      v19 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_30;
    }
LABEL_34:
    v20 = 2 * v5;
    goto LABEL_35;
  }
LABEL_3:
  v9[1] = a1;
  while ( 1 )
  {
    v22[0] = v10;
    v12 = *(__int64 **)(a1 + 40);
    if ( v12 == *(__int64 **)(a1 + 48) )
    {
      sub_1292090(a1 + 32, *(_BYTE **)(a1 + 40), v22);
      v10 = v22[0];
    }
    else
    {
      if ( v12 )
      {
        *v12 = v10;
        v12 = *(__int64 **)(a1 + 40);
      }
      *(_QWORD *)(a1 + 40) = v12 + 1;
    }
    result = *(__int64 **)(a1 + 64);
    if ( *(__int64 **)(a1 + 72) != result )
    {
LABEL_4:
      result = (__int64 *)sub_16CCBA0(a1 + 56, v10);
      goto LABEL_5;
    }
    v13 = &result[*(unsigned int *)(a1 + 84)];
    v14 = *(_DWORD *)(a1 + 84);
    if ( result == v13 )
    {
LABEL_22:
      if ( v14 >= *(_DWORD *)(a1 + 80) )
        goto LABEL_4;
      *(_DWORD *)(a1 + 84) = v14 + 1;
      *v13 = v10;
      ++*(_QWORD *)(a1 + 56);
    }
    else
    {
      v15 = 0;
      while ( v10 != *result )
      {
        if ( *result == -2 )
          v15 = result;
        if ( v13 == ++result )
        {
          if ( !v15 )
            goto LABEL_22;
          *v15 = v10;
          --*(_DWORD *)(a1 + 88);
          ++*(_QWORD *)(a1 + 56);
          a1 = *(_QWORD *)a1;
          if ( a1 )
            goto LABEL_6;
          return result;
        }
      }
    }
LABEL_5:
    a1 = *(_QWORD *)a1;
    if ( !a1 )
      return result;
LABEL_6:
    v10 = v21;
  }
}
