// Function: sub_9439D0
// Address: 0x9439d0
//
__int64 __fastcall sub_9439D0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  int v5; // r11d
  _QWORD *v6; // rdx
  unsigned int v7; // r8d
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 result; // rax
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  int v14; // esi
  __int64 v15; // r9
  unsigned int v16; // eax
  __int64 v17; // r8
  int v18; // r11d
  _QWORD *v19; // r10
  int v20; // eax
  int v21; // eax
  __int64 v22; // r8
  int v23; // r10d
  unsigned int v24; // r12d
  _QWORD *v25; // r9
  __int64 v26; // rsi

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = 1;
  v6 = 0;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
  {
LABEL_3:
    result = v8[1];
    if ( result )
    {
      if ( *(_DWORD *)(*(_QWORD *)(result + 8) + 8LL) >> 8 )
        return sub_92CAE0(a1, result, a2 + 64);
    }
    return result;
  }
  while ( v9 != -4096 )
  {
    if ( !v6 && v9 == -8192 )
      v6 = v8;
    v7 = (v3 - 1) & (v5 + v7);
    v8 = (_QWORD *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_3;
    ++v5;
  }
  if ( !v6 )
    v6 = v8;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_21:
    sub_9437F0(a1, 2 * v3);
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v6 = (_QWORD *)(v15 + 16LL * v16);
      v17 = *v6;
      if ( *v6 != a2 )
      {
        v18 = 1;
        v19 = 0;
        while ( v17 != -4096 )
        {
          if ( !v19 && v17 == -8192 )
            v19 = v6;
          v16 = v14 & (v18 + v16);
          v6 = (_QWORD *)(v15 + 16LL * v16);
          v17 = *v6;
          if ( *v6 == a2 )
            goto LABEL_17;
          ++v18;
        }
        if ( v19 )
          v6 = v19;
      }
      goto LABEL_17;
    }
    goto LABEL_44;
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_9437F0(a1, v3);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = 1;
      v24 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = 0;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v6 = (_QWORD *)(v22 + 16LL * v24);
      v26 = *v6;
      if ( *v6 != a2 )
      {
        while ( v26 != -4096 )
        {
          if ( !v25 && v26 == -8192 )
            v25 = v6;
          v24 = v21 & (v23 + v24);
          v6 = (_QWORD *)(v22 + 16LL * v24);
          v26 = *v6;
          if ( *v6 == a2 )
            goto LABEL_17;
          ++v23;
        }
        if ( v25 )
          v6 = v25;
      }
      goto LABEL_17;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v6 = a2;
  v6[1] = 0;
  return 0;
}
