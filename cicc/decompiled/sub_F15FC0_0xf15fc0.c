// Function: sub_F15FC0
// Address: 0xf15fc0
//
__int64 __fastcall sub_F15FC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  int v9; // r11d
  unsigned int v10; // edi
  __int64 result; // rax
  __int64 v12; // rcx
  int v13; // eax
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  unsigned int v17; // eax
  __int64 v18; // rdi
  int v19; // r10d
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdi
  unsigned int v23; // r13d
  __int64 v24; // rsi

  v2 = a1 + 2064;
  v5 = *(_DWORD *)(a1 + 2088);
  v6 = *(_DWORD *)(a1 + 8);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 2064);
    goto LABEL_19;
  }
  v7 = *(_QWORD *)(a1 + 2072);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v7 + 16LL * v10;
  v12 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    return result;
  while ( v12 != -4096 )
  {
    if ( v12 != -8192 || v8 )
      result = (__int64)v8;
    v10 = (v5 - 1) & (v9 + v10);
    v12 = *(_QWORD *)(v7 + 16LL * v10);
    if ( a2 == v12 )
      return result;
    ++v9;
    v8 = (_QWORD *)result;
    result = v7 + 16LL * v10;
  }
  if ( !v8 )
    v8 = (_QWORD *)result;
  v13 = *(_DWORD *)(a1 + 2080);
  ++*(_QWORD *)(a1 + 2064);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_19:
    sub_9BAAD0(v2, 2 * v5);
    v15 = *(_DWORD *)(a1 + 2088);
    if ( v15 )
    {
      v16 = v15 - 1;
      v7 = *(_QWORD *)(a1 + 2072);
      v17 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 2080) + 1;
      v8 = (_QWORD *)(v7 + 16LL * v17);
      v18 = *v8;
      if ( a2 != *v8 )
      {
        v19 = 1;
        v2 = 0;
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v2 )
            v2 = (__int64)v8;
          v17 = v16 & (v19 + v17);
          v8 = (_QWORD *)(v7 + 16LL * v17);
          v18 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          ++v19;
        }
        if ( v2 )
          v8 = (_QWORD *)v2;
      }
      goto LABEL_13;
    }
    goto LABEL_43;
  }
  if ( v5 - *(_DWORD *)(a1 + 2084) - v14 <= v5 >> 3 )
  {
    sub_9BAAD0(v2, v5);
    v20 = *(_DWORD *)(a1 + 2088);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 2072);
      v7 = 0;
      v23 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v2 = 1;
      v14 = *(_DWORD *)(a1 + 2080) + 1;
      v8 = (_QWORD *)(v22 + 16LL * v23);
      v24 = *v8;
      if ( a2 != *v8 )
      {
        while ( v24 != -4096 )
        {
          if ( !v7 && v24 == -8192 )
            v7 = (__int64)v8;
          v23 = v21 & (v2 + v23);
          v8 = (_QWORD *)(v22 + 16LL * v23);
          v24 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          v2 = (unsigned int)(v2 + 1);
        }
        if ( v7 )
          v8 = (_QWORD *)v7;
      }
      goto LABEL_13;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 2080);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 2080) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 2084);
  *v8 = a2;
  *((_DWORD *)v8 + 2) = v6;
  result = *(unsigned int *)(a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 8u, v7, v2);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
