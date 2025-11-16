// Function: sub_37AF270
// Address: 0x37af270
//
unsigned __int8 *__fastcall sub_37AF270(__int64 a1, unsigned __int64 a2, __int64 a3, __m128i a4)
{
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // r15d
  __int64 v10; // r14
  int v11; // r12d
  char v12; // dl
  __int64 v13; // r8
  int v14; // esi
  unsigned int v15; // ecx
  _DWORD *v16; // rax
  int v17; // r9d
  int *v18; // r12
  __int64 v19; // rdi
  int v20; // r8d
  unsigned int v21; // ecx
  __int64 v22; // rax
  int v23; // esi
  unsigned __int8 *v24; // r12
  __int64 v26; // rax
  unsigned int v27; // esi
  unsigned int v28; // eax
  _DWORD *v29; // rdi
  int v30; // ecx
  unsigned int v31; // r9d
  __int64 v32; // rax
  int v33; // eax
  int v34; // r10d
  __int64 v35; // rsi
  int v36; // eax
  unsigned int v37; // edx
  int v38; // ecx
  __int64 v39; // rsi
  int v40; // eax
  unsigned int v41; // edx
  int v42; // ecx
  int v43; // r9d
  _DWORD *v44; // r8
  int v45; // eax
  int v46; // eax
  int v47; // r10d
  int v48; // r9d
  __int64 v49; // [rsp+0h] [rbp-40h] BYREF
  int v50; // [rsp+8h] [rbp-38h]

  v7 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v49 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v49, v8, 1);
  v50 = *(_DWORD *)(a2 + 72);
  v11 = sub_375D5B0(a1, a2, a3);
  v12 = *(_BYTE *)(a1 + 720) & 1;
  if ( v12 )
  {
    v13 = a1 + 728;
    v14 = 7;
  }
  else
  {
    v27 = *(_DWORD *)(a1 + 736);
    v13 = *(_QWORD *)(a1 + 728);
    if ( !v27 )
    {
      v28 = *(_DWORD *)(a1 + 720);
      ++*(_QWORD *)(a1 + 712);
      v29 = 0;
      v30 = (v28 >> 1) + 1;
LABEL_18:
      v31 = 3 * v27;
      goto LABEL_19;
    }
    v14 = v27 - 1;
  }
  v15 = v14 & (37 * v11);
  v16 = (_DWORD *)(v13 + 8LL * v15);
  v17 = *v16;
  if ( v11 == *v16 )
  {
LABEL_6:
    v18 = v16 + 1;
    goto LABEL_7;
  }
  v34 = 1;
  v29 = 0;
  while ( v17 != -1 )
  {
    if ( !v29 && v17 == -2 )
      v29 = v16;
    v15 = v14 & (v34 + v15);
    v16 = (_DWORD *)(v13 + 8LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
      goto LABEL_6;
    ++v34;
  }
  v31 = 24;
  v27 = 8;
  if ( !v29 )
    v29 = v16;
  v28 = *(_DWORD *)(a1 + 720);
  ++*(_QWORD *)(a1 + 712);
  v30 = (v28 >> 1) + 1;
  if ( !v12 )
  {
    v27 = *(_DWORD *)(a1 + 736);
    goto LABEL_18;
  }
LABEL_19:
  if ( 4 * v30 >= v31 )
  {
    sub_375BDE0(a1 + 712, 2 * v27);
    if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
    {
      v35 = a1 + 728;
      v36 = 7;
    }
    else
    {
      v45 = *(_DWORD *)(a1 + 736);
      v35 = *(_QWORD *)(a1 + 728);
      if ( !v45 )
        goto LABEL_71;
      v36 = v45 - 1;
    }
    v37 = v36 & (37 * v11);
    v29 = (_DWORD *)(v35 + 8LL * v37);
    v38 = *v29;
    if ( v11 != *v29 )
    {
      v48 = 1;
      v44 = 0;
      while ( v38 != -1 )
      {
        if ( v38 == -2 && !v44 )
          v44 = v29;
        v37 = v36 & (v48 + v37);
        v29 = (_DWORD *)(v35 + 8LL * v37);
        v38 = *v29;
        if ( v11 == *v29 )
          goto LABEL_40;
        ++v48;
      }
      goto LABEL_46;
    }
LABEL_40:
    v28 = *(_DWORD *)(a1 + 720);
    goto LABEL_21;
  }
  if ( v27 - *(_DWORD *)(a1 + 724) - v30 <= v27 >> 3 )
  {
    sub_375BDE0(a1 + 712, v27);
    if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
    {
      v39 = a1 + 728;
      v40 = 7;
      goto LABEL_43;
    }
    v46 = *(_DWORD *)(a1 + 736);
    v39 = *(_QWORD *)(a1 + 728);
    if ( v46 )
    {
      v40 = v46 - 1;
LABEL_43:
      v41 = v40 & (37 * v11);
      v29 = (_DWORD *)(v39 + 8LL * v41);
      v42 = *v29;
      if ( v11 != *v29 )
      {
        v43 = 1;
        v44 = 0;
        while ( v42 != -1 )
        {
          if ( !v44 && v42 == -2 )
            v44 = v29;
          v41 = v40 & (v43 + v41);
          v29 = (_DWORD *)(v39 + 8LL * v41);
          v42 = *v29;
          if ( v11 == *v29 )
            goto LABEL_40;
          ++v43;
        }
LABEL_46:
        if ( v44 )
          v29 = v44;
        goto LABEL_40;
      }
      goto LABEL_40;
    }
LABEL_71:
    *(_DWORD *)(a1 + 720) = (2 * (*(_DWORD *)(a1 + 720) >> 1) + 2) | *(_DWORD *)(a1 + 720) & 1;
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a1 + 720) = (2 * (v28 >> 1) + 2) | v28 & 1;
  if ( *v29 != -1 )
    --*(_DWORD *)(a1 + 724);
  *v29 = v11;
  v18 = v29 + 1;
  v29[1] = 0;
LABEL_7:
  sub_37593F0(a1, v18);
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v19 = a1 + 520;
    v20 = 7;
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 528);
    v19 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v26 )
      goto LABEL_25;
    v20 = v26 - 1;
  }
  v21 = v20 & (37 * *v18);
  v22 = v19 + 24LL * v21;
  v23 = *(_DWORD *)v22;
  if ( *v18 == *(_DWORD *)v22 )
    goto LABEL_10;
  v33 = 1;
  while ( v23 != -1 )
  {
    v47 = v33 + 1;
    v21 = v20 & (v33 + v21);
    v22 = v19 + 24LL * v21;
    v23 = *(_DWORD *)v22;
    if ( *v18 == *(_DWORD *)v22 )
      goto LABEL_10;
    v33 = v47;
  }
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v32 = 192;
    goto LABEL_26;
  }
  v26 = *(unsigned int *)(a1 + 528);
LABEL_25:
  v32 = 24 * v26;
LABEL_26:
  v22 = v19 + v32;
LABEL_10:
  v24 = sub_34070B0(
          *(_QWORD **)(a1 + 8),
          *(_QWORD *)(v22 + 8),
          *(unsigned int *)(v22 + 16) | a3 & 0xFFFFFFFF00000000LL,
          (__int64)&v49,
          v9,
          v10,
          a4);
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v24;
}
