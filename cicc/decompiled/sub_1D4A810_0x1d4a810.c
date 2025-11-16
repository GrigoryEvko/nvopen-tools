// Function: sub_1D4A810
// Address: 0x1d4a810
//
__int64 __fastcall sub_1D4A810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v6; // r8d
  int v7; // r9d
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 k; // r14
  __int64 v13; // r14
  __int64 m; // r15
  __int64 v15; // rbx
  __int64 n; // r13
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  _QWORD *j; // rdx
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  int v22; // eax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  int v25; // ebx
  __int64 v26; // r14
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *i; // rdx
  _QWORD *v30; // rax

  result = *(_QWORD *)(*(_QWORD *)a2 + 1160LL);
  if ( (__int64 (*)())result == sub_1D45FE0 )
    return result;
  result = ((__int64 (__fastcall *)(__int64))result)(a2);
  if ( !(_BYTE)result )
    return result;
  *(_DWORD *)(a3 + 192) = 0;
  sub_1D4A640(a3 + 80);
  sub_1D4A640(a3 + 112);
  v8 = *(_DWORD *)(a3 + 160);
  ++*(_QWORD *)(a3 + 144);
  if ( v8 )
  {
    v17 = 4 * v8;
    a2 = 64;
    v9 = *(unsigned int *)(a3 + 168);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v17 = 64;
    if ( (unsigned int)v9 <= v17 )
      goto LABEL_32;
    v20 = *(_QWORD **)(a3 + 152);
    v21 = v8 - 1;
    if ( v21 )
    {
      _BitScanReverse(&v21, v21);
      v22 = 1 << (33 - (v21 ^ 0x1F));
      if ( v22 < 64 )
        v22 = 64;
      if ( (_DWORD)v9 == v22 )
      {
        *(_QWORD *)(a3 + 160) = 0;
        v30 = &v20[2 * (unsigned int)v9];
        do
        {
          if ( v20 )
            *v20 = -4;
          v20 += 2;
        }
        while ( v30 != v20 );
        goto LABEL_7;
      }
      v23 = (4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1);
      v24 = ((v23 | (v23 >> 2)) >> 4) | v23 | (v23 >> 2) | ((((v23 | (v23 >> 2)) >> 4) | v23 | (v23 >> 2)) >> 8);
      v25 = (v24 | (v24 >> 16)) + 1;
      v26 = 16 * ((v24 | (v24 >> 16)) + 1);
    }
    else
    {
      v26 = 2048;
      v25 = 128;
    }
    j___libc_free_0(v20);
    *(_DWORD *)(a3 + 168) = v25;
    v27 = (_QWORD *)sub_22077B0(v26);
    v28 = *(unsigned int *)(a3 + 168);
    *(_QWORD *)(a3 + 160) = 0;
    *(_QWORD *)(a3 + 152) = v27;
    for ( i = &v27[2 * v28]; i != v27; v27 += 2 )
    {
      if ( v27 )
        *v27 = -4;
    }
  }
  else if ( *(_DWORD *)(a3 + 164) )
  {
    v9 = *(unsigned int *)(a3 + 168);
    if ( (unsigned int)v9 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a3 + 152));
      *(_QWORD *)(a3 + 152) = 0;
      *(_QWORD *)(a3 + 160) = 0;
      *(_DWORD *)(a3 + 168) = 0;
      goto LABEL_7;
    }
LABEL_32:
    v18 = *(_QWORD **)(a3 + 152);
    for ( j = &v18[2 * v9]; j != v18; v18 += 2 )
      *v18 = -4;
    *(_QWORD *)(a3 + 160) = 0;
  }
LABEL_7:
  *(_QWORD *)(a3 + 176) = 0;
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, a2);
    v10 = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      sub_15E08E0(a1, a2);
    v11 = *(_QWORD *)(a1 + 88);
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 88);
    v11 = v10;
  }
  result = 5LL * *(_QWORD *)(a1 + 96);
  for ( k = v11 + 40LL * *(_QWORD *)(a1 + 96); v10 != k; ++*(_DWORD *)(a3 + 192) )
  {
    while ( 1 )
    {
      result = sub_15E02D0(v10);
      if ( (_BYTE)result )
        break;
      v10 += 40;
      if ( v10 == k )
        goto LABEL_16;
    }
    *(_QWORD *)(a3 + 176) = v10;
    result = *(unsigned int *)(a3 + 192);
    if ( (unsigned int)result >= *(_DWORD *)(a3 + 196) )
    {
      sub_16CD150(a3 + 184, (const void *)(a3 + 200), 0, 8, v6, v7);
      result = *(unsigned int *)(a3 + 192);
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 184) + 8 * result) = v10;
    v10 += 40;
  }
LABEL_16:
  v13 = *(_QWORD *)(a1 + 80);
  for ( m = a1 + 72; m != v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    if ( !v13 )
      BUG();
    v15 = *(_QWORD *)(v13 + 24);
    for ( n = v13 + 16; n != v15; v15 = *(_QWORD *)(v15 + 8) )
    {
      while ( 1 )
      {
        if ( !v15 )
          BUG();
        if ( *(_BYTE *)(v15 - 8) == 53 && (*(_BYTE *)(v15 - 6) & 0x40) != 0 )
          break;
        v15 = *(_QWORD *)(v15 + 8);
        if ( n == v15 )
          goto LABEL_27;
      }
      result = *(unsigned int *)(a3 + 192);
      if ( (unsigned int)result >= *(_DWORD *)(a3 + 196) )
      {
        sub_16CD150(a3 + 184, (const void *)(a3 + 200), 0, 8, v6, v7);
        result = *(unsigned int *)(a3 + 192);
      }
      *(_QWORD *)(*(_QWORD *)(a3 + 184) + 8 * result) = v15 - 24;
      ++*(_DWORD *)(a3 + 192);
    }
LABEL_27:
    ;
  }
  return result;
}
