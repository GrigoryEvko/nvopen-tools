// Function: sub_B30D10
// Address: 0xb30d10
//
void __fastcall sub_B30D10(__int64 a1, __int64 a2, __int64 a3)
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
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // r10d
  _QWORD *v28; // r9
  int v29; // eax
  int v30; // eax
  int v31; // r9d
  _QWORD *v32; // r8
  __int64 v33; // rdi
  unsigned int v34; // r15d
  __int64 v35; // rsi

  if ( *(char *)(a1 + 33) >= 0 )
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
  v10 = *(_DWORD *)(*(_QWORD *)v8 + 3344LL);
  v11 = *(_QWORD *)v8 + 3320LL;
  if ( !v10 )
  {
    ++*(_QWORD *)(v9 + 3320);
    goto LABEL_24;
  }
  v12 = 1;
  v13 = *(_QWORD *)(v9 + 3328);
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
    v20 = *(_DWORD *)(v9 + 3336);
    ++*(_QWORD *)(v9 + 3320);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(v9 + 3340) - v21 > v10 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(v9 + 3336) = v21;
        if ( *v14 != -4096 )
          --*(_DWORD *)(v9 + 3340);
        *v14 = a1;
        v19 = v14 + 1;
        v14[1] = 0;
        v14[2] = 0;
        goto LABEL_8;
      }
      sub_B30870(v11, v10);
      v29 = *(_DWORD *)(v9 + 3344);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = 1;
        v32 = 0;
        v33 = *(_QWORD *)(v9 + 3328);
        v34 = v30 & v15;
        v21 = *(_DWORD *)(v9 + 3336) + 1;
        v14 = (_QWORD *)(v33 + 24LL * v34);
        v35 = *v14;
        if ( a1 != *v14 )
        {
          while ( v35 != -4096 )
          {
            if ( !v32 && v35 == -8192 )
              v32 = v14;
            v34 = v30 & (v31 + v34);
            v14 = (_QWORD *)(v33 + 24LL * v34);
            v35 = *v14;
            if ( a1 == *v14 )
              goto LABEL_20;
            ++v31;
          }
          if ( v32 )
            v14 = v32;
        }
        goto LABEL_20;
      }
LABEL_47:
      ++*(_DWORD *)(v9 + 3336);
      BUG();
    }
LABEL_24:
    sub_B30870(v11, 2 * v10);
    v22 = *(_DWORD *)(v9 + 3344);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v9 + 3328);
      v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v21 = *(_DWORD *)(v9 + 3336) + 1;
      v14 = (_QWORD *)(v24 + 24LL * v25);
      v26 = *v14;
      if ( a1 != *v14 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( !v28 && v26 == -8192 )
            v28 = v14;
          v25 = v23 & (v27 + v25);
          v14 = (_QWORD *)(v24 + 24LL * v25);
          v26 = *v14;
          if ( a1 == *v14 )
            goto LABEL_20;
          ++v27;
        }
        if ( v28 )
          v14 = v28;
      }
      goto LABEL_20;
    }
    goto LABEL_47;
  }
LABEL_7:
  v19 = v17 + 1;
LABEL_8:
  *v19 = v6;
  v19[1] = v7;
  *(_BYTE *)(a1 + 33) = ((v7 != 0) << 7) | *(_BYTE *)(a1 + 33) & 0x7F;
}
