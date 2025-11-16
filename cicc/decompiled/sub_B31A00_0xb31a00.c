// Function: sub_B31A00
// Address: 0xb31a00
//
void __fastcall sub_B31A00(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // esi
  __int64 v11; // r10
  int v12; // r11d
  __int64 v13; // r8
  _QWORD *v14; // rdx
  unsigned int v15; // r15d
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 *v19; // rax
  unsigned __int16 v20; // dx
  __int16 v21; // ax
  __int16 v22; // cx
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rdi
  int v30; // r10d
  _QWORD *v31; // r9
  int v32; // eax
  int v33; // eax
  int v34; // r9d
  _QWORD *v35; // r8
  __int64 v36; // rdi
  unsigned int v37; // r15d
  __int64 v38; // rsi

  if ( (*(_BYTE *)(a1 + 35) & 4) == 0 )
  {
    if ( !a3 )
      return;
    goto LABEL_3;
  }
  v6 = a2;
  v7 = a3;
  if ( a3 )
  {
LABEL_3:
    v5 = (_QWORD *)sub_BD5C60(a1, a2, a3);
    v6 = sub_C94910(*v5 + 2736LL, a2, a3);
    v7 = a3;
  }
  v8 = sub_BD5C60(a1, a2, a3);
  v9 = *(_QWORD *)v8;
  v10 = *(_DWORD *)(*(_QWORD *)v8 + 3312LL);
  v11 = *(_QWORD *)v8 + 3288LL;
  if ( !v10 )
  {
    ++*(_QWORD *)(v9 + 3288);
    goto LABEL_26;
  }
  v12 = 1;
  v13 = *(_QWORD *)(v9 + 3296);
  v14 = 0;
  v15 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v16 = (v10 - 1) & v15;
  v17 = (_QWORD *)(v13 + 24LL * ((v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4))));
  v18 = *v17;
  if ( a1 != *v17 )
  {
    while ( v18 != -4096 )
    {
      if ( !v14 && v18 == -8192 )
        v14 = v17;
      v16 = (v10 - 1) & (v12 + v16);
      v17 = (_QWORD *)(v13 + 24LL * v16);
      v18 = *v17;
      if ( a1 == *v17 )
        goto LABEL_7;
      ++v12;
    }
    if ( !v14 )
      v14 = v17;
    v23 = *(_DWORD *)(v9 + 3304);
    ++*(_QWORD *)(v9 + 3288);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(v9 + 3308) - v24 > v10 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(v9 + 3304) = v24;
        if ( *v14 != -4096 )
          --*(_DWORD *)(v9 + 3308);
        *v14 = a1;
        v19 = v14 + 1;
        v14[1] = 0;
        v14[2] = 0;
        goto LABEL_8;
      }
      sub_B31800(v11, v10);
      v32 = *(_DWORD *)(v9 + 3312);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = 1;
        v35 = 0;
        v36 = *(_QWORD *)(v9 + 3296);
        v37 = v33 & v15;
        v24 = *(_DWORD *)(v9 + 3304) + 1;
        v14 = (_QWORD *)(v36 + 24LL * v37);
        v38 = *v14;
        if ( a1 != *v14 )
        {
          while ( v38 != -4096 )
          {
            if ( !v35 && v38 == -8192 )
              v35 = v14;
            v37 = v33 & (v34 + v37);
            v14 = (_QWORD *)(v36 + 24LL * v37);
            v38 = *v14;
            if ( a1 == *v14 )
              goto LABEL_22;
            ++v34;
          }
          if ( v35 )
            v14 = v35;
        }
        goto LABEL_22;
      }
LABEL_49:
      ++*(_DWORD *)(v9 + 3304);
      BUG();
    }
LABEL_26:
    sub_B31800(v11, 2 * v10);
    v25 = *(_DWORD *)(v9 + 3312);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v9 + 3296);
      v28 = (v25 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v24 = *(_DWORD *)(v9 + 3304) + 1;
      v14 = (_QWORD *)(v27 + 24LL * v28);
      v29 = *v14;
      if ( a1 != *v14 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -4096 )
        {
          if ( !v31 && v29 == -8192 )
            v31 = v14;
          v28 = v26 & (v30 + v28);
          v14 = (_QWORD *)(v27 + 24LL * v28);
          v29 = *v14;
          if ( a1 == *v14 )
            goto LABEL_22;
          ++v30;
        }
        if ( v31 )
          v14 = v31;
      }
      goto LABEL_22;
    }
    goto LABEL_49;
  }
LABEL_7:
  v19 = v17 + 1;
LABEL_8:
  *v19 = v6;
  v19[1] = v7;
  v20 = *(_WORD *)(a1 + 34);
  v21 = (v20 >> 1) & 0x7DFF;
  LOBYTE(v22) = v21;
  if ( v7 )
  {
    HIBYTE(v22) = HIBYTE(v21) | 2;
    v21 = v22;
  }
  *(_WORD *)(a1 + 34) = v20 & 1 | (2 * v21);
}
