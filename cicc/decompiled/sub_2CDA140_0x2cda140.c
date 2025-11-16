// Function: sub_2CDA140
// Address: 0x2cda140
//
unsigned __int64 __fastcall sub_2CDA140(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // rsi
  unsigned __int64 result; // rax
  __int64 v10; // r12
  unsigned int v11; // esi
  __int64 v12; // r8
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  int v15; // r9d
  int v16; // r9d
  __int64 v17; // r10
  _QWORD *v18; // rdi
  int v19; // edx
  __int64 v20; // rsi
  int v21; // r10d
  int v22; // eax
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // r9
  int v26; // r8d
  unsigned int v27; // r13d
  __int64 v28; // rsi
  int v29; // r8d
  _QWORD *v30; // rcx

  if ( **(_BYTE **)a1 && **(_BYTE **)(a1 + 8) )
  {
    v10 = *(_QWORD *)(a1 + 16);
    v11 = *(_DWORD *)(v10 + 24);
    if ( v11 )
    {
      v12 = *(_QWORD *)(v10 + 8);
      result = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (_QWORD *)(v12 + 8 * result);
      v14 = *v13;
      if ( *v13 == a2 )
        return result;
      v21 = 1;
      v18 = 0;
      while ( v14 != -4096 )
      {
        if ( !v18 && v14 == -8192 )
          v18 = v13;
        result = (v11 - 1) & (v21 + (_DWORD)result);
        v13 = (_QWORD *)(v12 + 8LL * (unsigned int)result);
        v14 = *v13;
        if ( *v13 == a2 )
          return result;
        ++v21;
      }
      v22 = *(_DWORD *)(v10 + 16);
      if ( !v18 )
        v18 = v13;
      ++*(_QWORD *)v10;
      v19 = v22 + 1;
      if ( 4 * (v22 + 1) < 3 * v11 )
      {
        result = v11 - *(_DWORD *)(v10 + 20) - v19;
        if ( (unsigned int)result > v11 >> 3 )
        {
LABEL_17:
          *(_DWORD *)(v10 + 16) = v19;
          if ( *v18 != -4096 )
            --*(_DWORD *)(v10 + 20);
          *v18 = a2;
          return result;
        }
        sub_27D4930(v10, v11);
        v23 = *(_DWORD *)(v10 + 24);
        if ( v23 )
        {
          v24 = v23 - 1;
          v25 = *(_QWORD *)(v10 + 8);
          v26 = 1;
          v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v18 = (_QWORD *)(v25 + 8LL * v27);
          v28 = *v18;
          v19 = *(_DWORD *)(v10 + 16) + 1;
          result = 0;
          if ( *v18 != a2 )
          {
            while ( v28 != -4096 )
            {
              if ( !result && v28 == -8192 )
                result = (unsigned __int64)v18;
              v27 = v24 & (v26 + v27);
              v18 = (_QWORD *)(v25 + 8LL * v27);
              v28 = *v18;
              if ( *v18 == a2 )
                goto LABEL_17;
              ++v26;
            }
            if ( result )
              v18 = (_QWORD *)result;
          }
          goto LABEL_17;
        }
LABEL_52:
        ++*(_DWORD *)(v10 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v10;
    }
    sub_27D4930(v10, 2 * v11);
    v15 = *(_DWORD *)(v10 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v10 + 8);
      result = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (_QWORD *)(v17 + 8 * result);
      v19 = *(_DWORD *)(v10 + 16) + 1;
      v20 = *v18;
      if ( *v18 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v20 != -4096 )
        {
          if ( !v30 && v20 == -8192 )
            v30 = v18;
          result = v16 & (unsigned int)(v29 + result);
          v18 = (_QWORD *)(v17 + 8LL * (unsigned int)result);
          v20 = *v18;
          if ( *v18 == a2 )
            goto LABEL_17;
          ++v29;
        }
        if ( v30 )
          v18 = v30;
      }
      goto LABEL_17;
    }
    goto LABEL_52;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v7 = **(_QWORD **)(a1 + 32);
  if ( !*(_BYTE *)(v6 + 28) )
    return (unsigned __int64)sub_C8CC70(*(_QWORD *)(a1 + 24), v7, (__int64)a3, a4, v6, a6);
  result = *(_QWORD *)(v6 + 8);
  a4 = *(unsigned int *)(v6 + 20);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_5:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
    {
      *(_DWORD *)(v6 + 20) = a4 + 1;
      *a3 = v7;
      ++*(_QWORD *)v6;
      return result;
    }
    return (unsigned __int64)sub_C8CC70(*(_QWORD *)(a1 + 24), v7, (__int64)a3, a4, v6, a6);
  }
  while ( v7 != *(_QWORD *)result )
  {
    result += 8LL;
    if ( a3 == (__int64 *)result )
      goto LABEL_5;
  }
  return result;
}
