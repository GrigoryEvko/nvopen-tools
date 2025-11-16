// Function: sub_1624210
// Address: 0x1624210
//
_QWORD *__fastcall sub_1624210(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // rdi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r9
  int v11; // r11d
  _QWORD *v12; // r13
  int v13; // eax
  int v14; // edx
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rsi
  int v27; // r9d
  _QWORD *v28; // r8
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  unsigned int v33; // r14d
  _QWORD *v34; // rdi
  __int64 v35; // rcx

  v2 = sub_16498A0(a1);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(*(_QWORD *)v2 + 424LL);
  v5 = *(_QWORD *)v2 + 400LL;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 400);
    goto LABEL_29;
  }
  v6 = *(_QWORD *)(v3 + 408);
  v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a1 )
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -8 )
    {
      if ( v9 == -16 && !v12 )
        v12 = v8;
      v7 = (v4 - 1) & (v11 + v7);
      v8 = (_QWORD *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a1 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    v13 = *(_DWORD *)(v3 + 416);
    ++*(_QWORD *)(v3 + 400);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(v3 + 420) - v14 > v4 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(v3 + 416) = v14;
        if ( *v12 != -8 )
          --*(_DWORD *)(v3 + 420);
        *v12 = a1;
        v12[1] = 0;
        goto LABEL_14;
      }
      sub_1624050(v5, v4);
      v29 = *(_DWORD *)(v3 + 424);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(v3 + 408);
        v32 = 1;
        v33 = v30 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v14 = *(_DWORD *)(v3 + 416) + 1;
        v34 = 0;
        v12 = (_QWORD *)(v31 + 16LL * v33);
        v35 = *v12;
        if ( *v12 != a1 )
        {
          while ( v35 != -8 )
          {
            if ( v35 == -16 && !v34 )
              v34 = v12;
            v33 = v30 & (v32 + v33);
            v12 = (_QWORD *)(v31 + 16LL * v33);
            v35 = *v12;
            if ( *v12 == a1 )
              goto LABEL_11;
            ++v32;
          }
          if ( v34 )
            v12 = v34;
        }
        goto LABEL_11;
      }
LABEL_58:
      ++*(_DWORD *)(v3 + 416);
      BUG();
    }
LABEL_29:
    sub_1624050(v5, 2 * v4);
    v22 = *(_DWORD *)(v3 + 424);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v3 + 408);
      v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = *(_DWORD *)(v3 + 416) + 1;
      v12 = (_QWORD *)(v24 + 16LL * v25);
      v26 = *v12;
      if ( *v12 != a1 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -8 )
        {
          if ( !v28 && v26 == -16 )
            v28 = v12;
          v25 = v23 & (v27 + v25);
          v12 = (_QWORD *)(v24 + 16LL * v25);
          v26 = *v12;
          if ( *v12 == a1 )
            goto LABEL_11;
          ++v27;
        }
        if ( v28 )
          v12 = v28;
      }
      goto LABEL_11;
    }
    goto LABEL_58;
  }
LABEL_3:
  if ( v8[1] )
    return (_QWORD *)v8[1];
  v12 = v8;
LABEL_14:
  *(_BYTE *)(a1 + 23) |= 0x10u;
  if ( *(_BYTE *)(a1 + 16) > 0x10u )
  {
    v19 = (_QWORD *)sub_22077B0(144);
    v16 = v19;
    if ( !v19 )
      goto LABEL_21;
    *v19 = 2;
    v20 = sub_16498A0(a1);
    v16[2] = 0;
    v16[3] = 0;
    v16[4] = 1;
    v16[1] = v20;
    v21 = v16 + 5;
    do
    {
      if ( v21 )
        *v21 = -4;
      v21 += 3;
    }
    while ( v16 + 17 != v21 );
    goto LABEL_20;
  }
  v15 = (_QWORD *)sub_22077B0(144);
  v16 = v15;
  if ( v15 )
  {
    *v15 = 1;
    v17 = sub_16498A0(a1);
    v16[2] = 0;
    v16[3] = 0;
    v16[4] = 1;
    v16[1] = v17;
    v18 = v16 + 5;
    do
    {
      if ( v18 )
        *v18 = -4;
      v18 += 3;
    }
    while ( v16 + 17 != v18 );
LABEL_20:
    v16[17] = a1;
  }
LABEL_21:
  v12[1] = v16;
  return v16;
}
