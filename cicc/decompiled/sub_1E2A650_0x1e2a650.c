// Function: sub_1E2A650
// Address: 0x1e2a650
//
__int64 *__fastcall sub_1E2A650(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 *result; // rax
  _BYTE *v12; // rsi
  __int64 v13; // rsi
  __int64 *v14; // rdi
  unsigned int v15; // r8d
  __int64 *v16; // rcx
  int v17; // r10d
  __int64 *v18; // r9
  int v19; // ecx
  int v20; // edx
  int v21; // r8d
  int v22; // r8d
  __int64 v23; // r10
  unsigned int v24; // ecx
  __int64 v25; // r9
  int v26; // edi
  __int64 *v27; // rsi
  int v28; // r8d
  int v29; // r8d
  __int64 v30; // r9
  __int64 *v31; // rdi
  __int64 v32; // r14
  int v33; // ecx
  __int64 v34; // rsi
  __int64 v35[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_DWORD *)(a3 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_33;
  }
  v7 = *(_QWORD *)(a3 + 8);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    v17 = 1;
    v18 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v18 )
        v18 = v9;
      v8 = (v6 - 1) & (v17 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_3;
      ++v17;
    }
    v19 = *(_DWORD *)(a3 + 16);
    if ( v18 )
      v9 = v18;
    ++*(_QWORD *)a3;
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a3 + 20) - v20 > v6 >> 3 )
      {
LABEL_29:
        *(_DWORD *)(a3 + 16) = v20;
        if ( *v9 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v9 = a2;
        v9[1] = 0;
        goto LABEL_3;
      }
      sub_1E2A490(a3, v6);
      v28 = *(_DWORD *)(a3 + 24);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a3 + 8);
        v31 = 0;
        LODWORD(v32) = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = *(_DWORD *)(a3 + 16) + 1;
        v33 = 1;
        v9 = (__int64 *)(v30 + 16LL * (unsigned int)v32);
        v34 = *v9;
        if ( *v9 != a2 )
        {
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v31 )
              v31 = v9;
            v32 = v29 & (unsigned int)(v32 + v33);
            v9 = (__int64 *)(v30 + 16 * v32);
            v34 = *v9;
            if ( *v9 == a2 )
              goto LABEL_29;
            ++v33;
          }
          if ( v31 )
            v9 = v31;
        }
        goto LABEL_29;
      }
LABEL_61:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_33:
    sub_1E2A490(a3, 2 * v6);
    v21 = *(_DWORD *)(a3 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a3 + 8);
      v24 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = *(_DWORD *)(a3 + 16) + 1;
      v9 = (__int64 *)(v23 + 16LL * v24);
      v25 = *v9;
      if ( *v9 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -8 )
        {
          if ( !v27 && v25 == -16 )
            v27 = v9;
          v24 = v22 & (v26 + v24);
          v9 = (__int64 *)(v23 + 16LL * v24);
          v25 = *v9;
          if ( *v9 == a2 )
            goto LABEL_29;
          ++v26;
        }
        if ( v27 )
          v9 = v27;
      }
      goto LABEL_29;
    }
    goto LABEL_61;
  }
LABEL_3:
  v9[1] = (__int64)a1;
  do
  {
LABEL_6:
    v35[0] = a2;
    v12 = (_BYTE *)a1[5];
    if ( v12 == (_BYTE *)a1[6] )
    {
      sub_1D4AF10((__int64)(a1 + 4), v12, v35);
      v13 = v35[0];
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = a2;
        v12 = (_BYTE *)a1[5];
      }
      a1[5] = (__int64)(v12 + 8);
      v13 = a2;
    }
    result = (__int64 *)a1[8];
    if ( (__int64 *)a1[9] != result )
    {
LABEL_4:
      result = sub_16CCBA0((__int64)(a1 + 7), v13);
      goto LABEL_5;
    }
    v14 = &result[*((unsigned int *)a1 + 21)];
    v15 = *((_DWORD *)a1 + 21);
    if ( result == v14 )
    {
LABEL_21:
      if ( v15 >= *((_DWORD *)a1 + 20) )
        goto LABEL_4;
      *((_DWORD *)a1 + 21) = v15 + 1;
      *v14 = v13;
      ++a1[7];
    }
    else
    {
      v16 = 0;
      while ( v13 != *result )
      {
        if ( *result == -2 )
          v16 = result;
        if ( v14 == ++result )
        {
          if ( !v16 )
            goto LABEL_21;
          *v16 = v13;
          --*((_DWORD *)a1 + 22);
          ++a1[7];
          a1 = (__int64 *)*a1;
          if ( a1 )
            goto LABEL_6;
          return result;
        }
      }
    }
LABEL_5:
    a1 = (__int64 *)*a1;
  }
  while ( a1 );
  return result;
}
