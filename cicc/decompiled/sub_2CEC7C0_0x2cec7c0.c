// Function: sub_2CEC7C0
// Address: 0x2cec7c0
//
__int64 *__fastcall sub_2CEC7C0(__int64 a1, int a2, __int64 a3, _BYTE *a4)
{
  unsigned int v8; // esi
  __int64 v9; // rdx
  unsigned int v10; // r8d
  unsigned int v11; // r13d
  unsigned int v12; // edi
  __int64 *result; // rax
  __int64 v14; // rcx
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rsi
  unsigned int v18; // eax
  int v19; // edx
  __int64 *v20; // r9
  __int64 v21; // rdi
  int v22; // r10d
  __int64 *v23; // r8
  __int64 v24; // r9
  unsigned int v25; // r11d
  int i; // r10d
  __int64 *v27; // r10
  int v28; // r10d
  int v29; // eax
  int v30; // eax
  int v31; // eax
  __int64 v32; // rsi
  __int64 *v33; // rdi
  unsigned int v34; // r13d
  int v35; // r8d
  __int64 v36; // rcx
  int v37; // [rsp+Ch] [rbp-34h]

  v8 = *(_DWORD *)(a3 + 24);
  v9 = *(_QWORD *)(a3 + 8);
  if ( !v8 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_10;
  }
  v10 = v8 - 1;
  v11 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v12 = (v8 - 1) & v11;
  result = (__int64 *)(v9 + 16LL * v12);
  v14 = *result;
  if ( a1 != *result )
  {
    v24 = *result;
    v25 = (v8 - 1) & v11;
    for ( i = 1; ; i = v37 )
    {
      if ( v24 == -4096 )
        goto LABEL_5;
      v25 = v10 & (i + v25);
      v37 = i + 1;
      v27 = (__int64 *)(v9 + 16LL * v25);
      v24 = *v27;
      if ( a1 == *v27 )
        break;
    }
    if ( v27 == (__int64 *)(v9 + 16LL * v8) )
      goto LABEL_22;
    result = (__int64 *)(v9 + 16LL * v25);
LABEL_4:
    if ( *((_DWORD *)result + 2) == a2 )
      return result;
LABEL_5:
    v11 = ((unsigned int)a1 >> 4) ^ ((unsigned int)a1 >> 9);
    v12 = v10 & v11;
    result = (__int64 *)(v9 + 16LL * (v10 & v11));
    v14 = *result;
    if ( a1 == *result )
      goto LABEL_6;
LABEL_22:
    v28 = 1;
    v20 = 0;
    while ( v14 != -4096 )
    {
      if ( !v20 && v14 == -8192 )
        v20 = result;
      v12 = v10 & (v28 + v12);
      result = (__int64 *)(v9 + 16LL * v12);
      v14 = *result;
      if ( a1 == *result )
        goto LABEL_6;
      ++v28;
    }
    if ( !v20 )
      v20 = result;
    v29 = *(_DWORD *)(a3 + 16);
    ++*(_QWORD *)a3;
    v19 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a3 + 20) - v19 > v8 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(a3 + 16) = v19;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v20 = a1;
        result = v20 + 1;
        *((_DWORD *)v20 + 2) = 0;
        goto LABEL_7;
      }
      sub_D39D40(a3, v8);
      v30 = *(_DWORD *)(a3 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a3 + 8);
        v33 = 0;
        v34 = v31 & v11;
        v35 = 1;
        v19 = *(_DWORD *)(a3 + 16) + 1;
        v20 = (__int64 *)(v32 + 16LL * v34);
        v36 = *v20;
        if ( a1 != *v20 )
        {
          while ( v36 != -4096 )
          {
            if ( !v33 && v36 == -8192 )
              v33 = v20;
            v34 = v31 & (v35 + v34);
            v20 = (__int64 *)(v32 + 16LL * v34);
            v36 = *v20;
            if ( a1 == *v20 )
              goto LABEL_28;
            ++v35;
          }
          if ( v33 )
            v20 = v33;
        }
        goto LABEL_28;
      }
LABEL_53:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_10:
    sub_D39D40(a3, 2 * v8);
    v15 = *(_DWORD *)(a3 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a3 + 8);
      v18 = (v15 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v19 = *(_DWORD *)(a3 + 16) + 1;
      v20 = (__int64 *)(v17 + 16LL * v18);
      v21 = *v20;
      if ( a1 != *v20 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v23 )
            v23 = v20;
          v18 = v16 & (v22 + v18);
          v20 = (__int64 *)(v17 + 16LL * v18);
          v21 = *v20;
          if ( a1 == *v20 )
            goto LABEL_28;
          ++v22;
        }
        if ( v23 )
          v20 = v23;
      }
      goto LABEL_28;
    }
    goto LABEL_53;
  }
  if ( result != (__int64 *)(v9 + 16LL * v8) )
    goto LABEL_4;
LABEL_6:
  ++result;
LABEL_7:
  *(_DWORD *)result = a2;
  *a4 = 1;
  return result;
}
