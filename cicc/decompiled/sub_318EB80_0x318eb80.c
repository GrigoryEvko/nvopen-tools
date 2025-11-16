// Function: sub_318EB80
// Address: 0x318eb80
//
_QWORD *__fastcall sub_318EB80(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // r12
  unsigned int v3; // esi
  __int64 v4; // rdi
  int v5; // r11d
  __int64 v6; // rcx
  _QWORD *v7; // r14
  unsigned int v8; // edx
  _QWORD *v9; // rax
  _QWORD *v10; // r9
  int v12; // eax
  int v13; // edx
  _QWORD *v14; // rax
  unsigned __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r9d
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  _QWORD *v26; // rdi
  unsigned int v27; // r13d
  int v28; // r8d
  __int64 v29; // rcx

  v1 = *(_QWORD *)(a1 + 24);
  v2 = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 8LL);
  if ( !v2 )
    return v2;
  v3 = *(_DWORD *)(v1 + 176);
  v4 = v1 + 152;
  if ( v3 )
  {
    v5 = 1;
    v6 = *(_QWORD *)(v1 + 160);
    v7 = 0;
    v8 = (v3 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v9 = (_QWORD *)(v6 + 16LL * v8);
    v10 = (_QWORD *)*v9;
    if ( v2 == (_QWORD *)*v9 )
      return (_QWORD *)v9[1];
    while ( v10 != (_QWORD *)-4096LL )
    {
      if ( v10 == (_QWORD *)-8192LL && !v7 )
        v7 = v9;
      v8 = (v3 - 1) & (v5 + v8);
      v9 = (_QWORD *)(v6 + 16LL * v8);
      v10 = (_QWORD *)*v9;
      if ( v2 == (_QWORD *)*v9 )
        return (_QWORD *)v9[1];
      ++v5;
    }
    if ( !v7 )
      v7 = v9;
    v12 = *(_DWORD *)(v1 + 168);
    ++*(_QWORD *)(v1 + 152);
    v13 = v12 + 1;
    if ( 4 * (v12 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(v1 + 172) - v13 > v3 >> 3 )
        goto LABEL_16;
      sub_318CD80(v4, v3);
      v23 = *(_DWORD *)(v1 + 176);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(v1 + 160);
        v26 = 0;
        v27 = v24 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v28 = 1;
        v13 = *(_DWORD *)(v1 + 168) + 1;
        v7 = (_QWORD *)(v25 + 16LL * v27);
        v29 = *v7;
        if ( v2 != (_QWORD *)*v7 )
        {
          while ( v29 != -4096 )
          {
            if ( !v26 && v29 == -8192 )
              v26 = v7;
            v27 = v24 & (v28 + v27);
            v7 = (_QWORD *)(v25 + 16LL * v27);
            v29 = *v7;
            if ( v2 == (_QWORD *)*v7 )
              goto LABEL_16;
            ++v28;
          }
          if ( v26 )
            v7 = v26;
        }
        goto LABEL_16;
      }
LABEL_46:
      ++*(_DWORD *)(v1 + 168);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v1 + 152);
  }
  sub_318CD80(v4, 2 * v3);
  v16 = *(_DWORD *)(v1 + 176);
  if ( !v16 )
    goto LABEL_46;
  v17 = v16 - 1;
  v18 = *(_QWORD *)(v1 + 160);
  v19 = (v16 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v13 = *(_DWORD *)(v1 + 168) + 1;
  v7 = (_QWORD *)(v18 + 16LL * v19);
  v20 = *v7;
  if ( v2 != (_QWORD *)*v7 )
  {
    v21 = 1;
    v22 = 0;
    while ( v20 != -4096 )
    {
      if ( !v22 && v20 == -8192 )
        v22 = v7;
      v19 = v17 & (v21 + v19);
      v7 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v7;
      if ( v2 == (_QWORD *)*v7 )
        goto LABEL_16;
      ++v21;
    }
    if ( v22 )
      v7 = v22;
  }
LABEL_16:
  *(_DWORD *)(v1 + 168) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(v1 + 172);
  *v7 = v2;
  v7[1] = 0;
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
  {
    *v14 = v2;
    v14[1] = v1;
  }
  v15 = v7[1];
  v2 = v14;
  v7[1] = v14;
  if ( !v15 )
    return v2;
  j_j___libc_free_0(v15);
  return (_QWORD *)v7[1];
}
