// Function: sub_1CA7B50
// Address: 0x1ca7b50
//
__int64 *__fastcall sub_1CA7B50(__int64 a1, int a2, __int64 a3, _BYTE *a4)
{
  unsigned int v8; // esi
  __int64 v9; // rdx
  unsigned int v10; // r8d
  int v11; // r10d
  unsigned int v12; // r14d
  unsigned int v13; // edi
  unsigned int v14; // r11d
  __int64 *result; // rax
  __int64 v16; // rcx
  __int64 v17; // r9
  int v18; // eax
  int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // edx
  int v22; // ecx
  __int64 v23; // r8
  int v24; // r10d
  __int64 *v25; // r9
  __int64 *v26; // r10
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // edx
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  int v33; // r9d
  __int64 v34; // r14
  __int64 *v35; // r8
  __int64 v36; // rsi
  int v37; // [rsp+Ch] [rbp-34h]

  v8 = *(_DWORD *)(a3 + 24);
  v9 = *(_QWORD *)(a3 + 8);
  if ( !v8 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_9;
  }
  v10 = v8 - 1;
  v11 = 1;
  v12 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v13 = (v8 - 1) & v12;
  v14 = v13;
  result = (__int64 *)(v9 + 16LL * v13);
  v16 = *result;
  v17 = *result;
  if ( a1 == *result )
  {
    if ( result == (__int64 *)(v9 + 16LL * v8) )
      goto LABEL_6;
  }
  else
  {
    while ( 1 )
    {
      if ( v17 == -8 )
        goto LABEL_5;
      v14 = v10 & (v11 + v14);
      v37 = v11 + 1;
      v26 = (__int64 *)(v9 + 16LL * v14);
      v17 = *v26;
      if ( a1 == *v26 )
        break;
      v11 = v37;
    }
    if ( v26 == (__int64 *)(v9 + 16LL * v8) )
      goto LABEL_21;
    result = (__int64 *)(v9 + 16LL * v14);
  }
  if ( *((_DWORD *)result + 2) == a2 )
    return result;
LABEL_5:
  v12 = ((unsigned int)a1 >> 4) ^ ((unsigned int)a1 >> 9);
  v13 = v10 & v12;
  result = (__int64 *)(v9 + 16LL * (v10 & v12));
  v16 = *result;
  if ( a1 != *result )
  {
LABEL_21:
    v27 = 1;
    v28 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v28 )
        v28 = result;
      v13 = v10 & (v27 + v13);
      result = (__int64 *)(v9 + 16LL * v13);
      v16 = *result;
      if ( a1 == *result )
        goto LABEL_6;
      ++v27;
    }
    v29 = *(_DWORD *)(a3 + 16);
    if ( v28 )
      result = v28;
    ++*(_QWORD *)a3;
    v22 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a3 + 20) - v22 > v8 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a3 + 16) = v22;
        if ( *result != -8 )
          --*(_DWORD *)(a3 + 20);
        *result = a1;
        *((_DWORD *)result + 2) = 0;
        goto LABEL_6;
      }
      sub_177C7D0(a3, v8);
      v30 = *(_DWORD *)(a3 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a3 + 8);
        v33 = 1;
        LODWORD(v34) = (v30 - 1) & v12;
        v35 = 0;
        v22 = *(_DWORD *)(a3 + 16) + 1;
        result = (__int64 *)(v32 + 16LL * (unsigned int)v34);
        v36 = *result;
        if ( a1 != *result )
        {
          while ( v36 != -8 )
          {
            if ( !v35 && v36 == -16 )
              v35 = result;
            v34 = v31 & (unsigned int)(v34 + v33);
            result = (__int64 *)(v32 + 16 * v34);
            v36 = *result;
            if ( a1 == *result )
              goto LABEL_27;
            ++v33;
          }
          if ( v35 )
            result = v35;
        }
        goto LABEL_27;
      }
LABEL_51:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_9:
    sub_177C7D0(a3, 2 * v8);
    v18 = *(_DWORD *)(a3 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a3 + 8);
      v21 = (v18 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v22 = *(_DWORD *)(a3 + 16) + 1;
      result = (__int64 *)(v20 + 16LL * v21);
      v23 = *result;
      if ( a1 != *result )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( v23 == -16 && !v25 )
            v25 = result;
          v21 = v19 & (v24 + v21);
          result = (__int64 *)(v20 + 16LL * v21);
          v23 = *result;
          if ( a1 == *result )
            goto LABEL_27;
          ++v24;
        }
        if ( v25 )
          result = v25;
      }
      goto LABEL_27;
    }
    goto LABEL_51;
  }
LABEL_6:
  *((_DWORD *)result + 2) = a2;
  *a4 = 1;
  return result;
}
