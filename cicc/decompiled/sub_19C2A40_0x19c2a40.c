// Function: sub_19C2A40
// Address: 0x19c2a40
//
unsigned __int64 __fastcall sub_19C2A40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // rax
  unsigned __int64 result; // rax
  __int64 v12; // r8
  __int64 *v13; // rdi
  unsigned int v14; // r9d
  __int64 *v15; // rsi
  int v16; // r10d
  __int64 *v17; // r9
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  __int64 v23; // rdi
  int v24; // r10d
  __int64 *v25; // r9
  int v26; // eax
  __int64 v27; // rdi
  int v28; // r9d
  __int64 *v29; // r8
  unsigned int v30; // r13d
  __int64 v31; // rsi

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_26;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 112LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    v16 = 1;
    v17 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v17 )
        v17 = v9;
      v8 = (v5 - 1) & (v16 + v8);
      v9 = (__int64 *)(v7 + 112LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_3;
      ++v16;
    }
    v18 = *(_DWORD *)(a1 + 16);
    if ( v17 )
      v9 = v17;
    ++*(_QWORD *)a1;
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v5 )
    {
      result = v5 - *(_DWORD *)(a1 + 20) - v19;
      if ( (unsigned int)result > v5 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 16) = v19;
        if ( *v9 != -8 )
          --*(_DWORD *)(a1 + 20);
        v13 = v9 + 6;
        *v9 = a2;
        v12 = (__int64)(v9 + 1);
        v14 = 0;
        v9[1] = 0;
        v9[2] = (__int64)(v9 + 6);
        v9[3] = (__int64)(v9 + 6);
        v9[4] = 8;
        *((_DWORD *)v9 + 10) = 0;
        goto LABEL_23;
      }
      sub_19C24C0(a1, v5);
      v26 = *(_DWORD *)(a1 + 24);
      if ( v26 )
      {
        result = (unsigned int)(v26 - 1);
        v27 = *(_QWORD *)(a1 + 8);
        v28 = 1;
        v29 = 0;
        v30 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v9 = (__int64 *)(v27 + 112LL * v30);
        v19 = *(_DWORD *)(a1 + 16) + 1;
        v31 = *v9;
        if ( *v9 != a2 )
        {
          while ( v31 != -8 )
          {
            if ( v31 == -16 && !v29 )
              v29 = v9;
            v30 = result & (v28 + v30);
            v9 = (__int64 *)(v27 + 112LL * v30);
            v31 = *v9;
            if ( *v9 == a2 )
              goto LABEL_20;
            ++v28;
          }
          if ( v29 )
            v9 = v29;
        }
        goto LABEL_20;
      }
LABEL_54:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_26:
    sub_19C24C0(a1, 2 * v5);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      result = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v22 + 112 * result);
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v23 = *v9;
      if ( *v9 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v9;
          result = v21 & (unsigned int)(v24 + result);
          v9 = (__int64 *)(v22 + 112LL * (unsigned int)result);
          v23 = *v9;
          if ( *v9 == a2 )
            goto LABEL_20;
          ++v24;
        }
        if ( v25 )
          v9 = v25;
      }
      goto LABEL_20;
    }
    goto LABEL_54;
  }
LABEL_3:
  result = v9[2];
  v12 = (__int64)(v9 + 1);
  if ( v9[3] != result )
    return (unsigned __int64)sub_16CCBA0(v12, a3);
  v13 = (__int64 *)(result + 8LL * *((unsigned int *)v9 + 9));
  v14 = *((_DWORD *)v9 + 9);
  if ( v13 != (__int64 *)result )
  {
    v15 = 0;
    while ( a3 != *(_QWORD *)result )
    {
      if ( *(_QWORD *)result == -2 )
        v15 = (__int64 *)result;
      result += 8LL;
      if ( v13 == (__int64 *)result )
      {
        if ( !v15 )
          goto LABEL_23;
        *v15 = a3;
        --*((_DWORD *)v9 + 10);
        ++v9[1];
        return result;
      }
    }
    return result;
  }
LABEL_23:
  if ( v14 >= *((_DWORD *)v9 + 8) )
    return (unsigned __int64)sub_16CCBA0(v12, a3);
  *((_DWORD *)v9 + 9) = v14 + 1;
  *v13 = a3;
  ++v9[1];
  return result;
}
