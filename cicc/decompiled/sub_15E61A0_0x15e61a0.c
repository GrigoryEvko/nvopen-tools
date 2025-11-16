// Function: sub_15E61A0
// Address: 0x15e61a0
//
__int64 __fastcall sub_15E61A0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  int v11; // r11d
  _QWORD *v12; // r10
  int v13; // ecx
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // edx
  int v24; // r9d
  _QWORD *v25; // r8
  __int64 v26; // rdi
  unsigned int v27; // r13d
  __int64 v28; // rsi

  v2 = sub_16498A0(a1);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(*(_QWORD *)v2 + 2792LL);
  v5 = *(_QWORD *)v2 + 2768LL;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 2768);
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(v3 + 2776);
  v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (_QWORD *)(v6 + 24LL * v7);
  v9 = *v8;
  if ( a1 == *v8 )
    return v8[1];
  v11 = 1;
  v12 = 0;
  while ( v9 != -8 )
  {
    if ( !v12 && v9 == -16 )
      v12 = v8;
    v7 = (v4 - 1) & (v11 + v7);
    v8 = (_QWORD *)(v6 + 24LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
      return v8[1];
    ++v11;
  }
  v13 = *(_DWORD *)(v3 + 2784);
  if ( v12 )
    v8 = v12;
  ++*(_QWORD *)(v3 + 2768);
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v4 )
  {
LABEL_14:
    sub_15E5B50(v5, 2 * v4);
    v15 = *(_DWORD *)(v3 + 2792);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v3 + 2776);
      v18 = (v15 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = *(_DWORD *)(v3 + 2784) + 1;
      v8 = (_QWORD *)(v17 + 24LL * v18);
      v19 = *v8;
      if ( a1 != *v8 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -8 )
        {
          if ( !v21 && v19 == -16 )
            v21 = v8;
          v18 = v16 & (v20 + v18);
          v8 = (_QWORD *)(v17 + 24LL * v18);
          v19 = *v8;
          if ( a1 == *v8 )
            goto LABEL_10;
          ++v20;
        }
        if ( v21 )
          v8 = v21;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  if ( v4 - *(_DWORD *)(v3 + 2788) - v14 <= v4 >> 3 )
  {
    sub_15E5B50(v5, v4);
    v22 = *(_DWORD *)(v3 + 2792);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = 1;
      v25 = 0;
      v26 = *(_QWORD *)(v3 + 2776);
      v27 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = *(_DWORD *)(v3 + 2784) + 1;
      v8 = (_QWORD *)(v26 + 24LL * v27);
      v28 = *v8;
      if ( a1 != *v8 )
      {
        while ( v28 != -8 )
        {
          if ( !v25 && v28 == -16 )
            v25 = v8;
          v27 = v23 & (v24 + v27);
          v8 = (_QWORD *)(v26 + 24LL * v27);
          v28 = *v8;
          if ( a1 == *v8 )
            goto LABEL_10;
          ++v24;
        }
        if ( v25 )
          v8 = v25;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(v3 + 2784);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(v3 + 2784) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(v3 + 2788);
  *v8 = a1;
  v8[1] = 0;
  v8[2] = 0;
  return v8[1];
}
