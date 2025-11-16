// Function: sub_1C8ED80
// Address: 0x1c8ed80
//
unsigned __int64 __fastcall sub_1C8ED80(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rsi
  unsigned __int64 result; // rax
  __int64 *v5; // rdi
  unsigned int v6; // r9d
  __int64 *v7; // rcx
  __int64 v9; // r12
  unsigned int v10; // esi
  __int64 v11; // r8
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  int v14; // r9d
  int v15; // r9d
  __int64 v16; // r10
  _QWORD *v17; // rdi
  int v18; // edx
  __int64 v19; // rsi
  int v20; // r10d
  int v21; // eax
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // r9
  int v25; // r8d
  unsigned int v26; // r13d
  __int64 v27; // rsi
  int v28; // r8d
  _QWORD *v29; // rcx

  if ( **(_BYTE **)a1 && **(_BYTE **)(a1 + 8) )
  {
    v9 = *(_QWORD *)(a1 + 16);
    v10 = *(_DWORD *)(v9 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(v9 + 8);
      result = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (_QWORD *)(v11 + 8 * result);
      v13 = *v12;
      if ( *v12 == a2 )
        return result;
      v20 = 1;
      v17 = 0;
      while ( v13 != -8 )
      {
        if ( !v17 && v13 == -16 )
          v17 = v12;
        result = (v10 - 1) & (v20 + (_DWORD)result);
        v12 = (_QWORD *)(v11 + 8LL * (unsigned int)result);
        v13 = *v12;
        if ( *v12 == a2 )
          return result;
        ++v20;
      }
      v21 = *(_DWORD *)(v9 + 16);
      if ( !v17 )
        v17 = v12;
      ++*(_QWORD *)v9;
      v18 = v21 + 1;
      if ( 4 * (v21 + 1) < 3 * v10 )
      {
        result = v10 - *(_DWORD *)(v9 + 20) - v18;
        if ( (unsigned int)result > v10 >> 3 )
        {
LABEL_22:
          *(_DWORD *)(v9 + 16) = v18;
          if ( *v17 != -8 )
            --*(_DWORD *)(v9 + 20);
          *v17 = a2;
          return result;
        }
        sub_1C8EBD0(v9, v10);
        v22 = *(_DWORD *)(v9 + 24);
        if ( v22 )
        {
          v23 = v22 - 1;
          v24 = *(_QWORD *)(v9 + 8);
          v25 = 1;
          v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v17 = (_QWORD *)(v24 + 8LL * v26);
          v27 = *v17;
          v18 = *(_DWORD *)(v9 + 16) + 1;
          result = 0;
          if ( *v17 != a2 )
          {
            while ( v27 != -8 )
            {
              if ( !result && v27 == -16 )
                result = (unsigned __int64)v17;
              v26 = v23 & (v25 + v26);
              v17 = (_QWORD *)(v24 + 8LL * v26);
              v27 = *v17;
              if ( *v17 == a2 )
                goto LABEL_22;
              ++v25;
            }
            if ( result )
              v17 = (_QWORD *)result;
          }
          goto LABEL_22;
        }
LABEL_57:
        ++*(_DWORD *)(v9 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v9;
    }
    sub_1C8EBD0(v9, 2 * v10);
    v14 = *(_DWORD *)(v9 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v9 + 8);
      result = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = (_QWORD *)(v16 + 8 * result);
      v18 = *(_DWORD *)(v9 + 16) + 1;
      v19 = *v17;
      if ( *v17 != a2 )
      {
        v28 = 1;
        v29 = 0;
        while ( v19 != -8 )
        {
          if ( !v29 && v19 == -16 )
            v29 = v17;
          result = v15 & (unsigned int)(v28 + result);
          v17 = (_QWORD *)(v16 + 8LL * (unsigned int)result);
          v19 = *v17;
          if ( *v17 == a2 )
            goto LABEL_22;
          ++v28;
        }
        if ( v29 )
          v17 = v29;
      }
      goto LABEL_22;
    }
    goto LABEL_57;
  }
  v2 = *(_QWORD *)(a1 + 24);
  v3 = **(_QWORD **)(a1 + 32);
  result = *(_QWORD *)(v2 + 8);
  if ( *(_QWORD *)(v2 + 16) != result )
    return (unsigned __int64)sub_16CCBA0(v2, v3);
  v5 = (__int64 *)(result + 8LL * *(unsigned int *)(v2 + 28));
  v6 = *(_DWORD *)(v2 + 28);
  if ( (__int64 *)result == v5 )
  {
LABEL_17:
    if ( v6 >= *(_DWORD *)(v2 + 24) )
      return (unsigned __int64)sub_16CCBA0(v2, v3);
    *(_DWORD *)(v2 + 28) = v6 + 1;
    *v5 = v3;
    ++*(_QWORD *)v2;
  }
  else
  {
    v7 = 0;
    while ( v3 != *(_QWORD *)result )
    {
      if ( *(_QWORD *)result == -2 )
        v7 = (__int64 *)result;
      result += 8LL;
      if ( v5 == (__int64 *)result )
      {
        if ( !v7 )
          goto LABEL_17;
        *v7 = v3;
        --*(_DWORD *)(v2 + 32);
        ++*(_QWORD *)v2;
        return result;
      }
    }
  }
  return result;
}
