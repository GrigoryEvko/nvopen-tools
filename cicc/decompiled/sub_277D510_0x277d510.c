// Function: sub_277D510
// Address: 0x277d510
//
__int64 __fastcall sub_277D510(__int64 a1, __int64 a2, int *a3, int a4)
{
  __int64 v4; // r9
  int v6; // edx
  int v7; // eax
  int v9; // r10d
  __int64 v10; // r13
  unsigned int v12; // eax
  __int64 v13; // r11
  char v14; // al
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // r15
  __int64 v18; // rax
  char v19; // al
  __int64 v20; // r15
  unsigned __int16 v21; // ax
  bool v22; // al
  char v23; // al
  int v24; // r10d
  __m128i v25; // xmm1
  __m128i v26; // xmm2
  _QWORD *v27; // rax
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rax
  char v39; // al
  unsigned __int16 v40; // ax
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // eax
  unsigned int v44; // [rsp+4h] [rbp-BCh]
  int v46; // [rsp+8h] [rbp-B8h]
  __int64 v47; // [rsp+8h] [rbp-B8h]
  int v48; // [rsp+8h] [rbp-B8h]
  int v49; // [rsp+8h] [rbp-B8h]
  int v50; // [rsp+8h] [rbp-B8h]
  __int64 v51; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v52; // [rsp+10h] [rbp-B0h]
  __int64 v53; // [rsp+10h] [rbp-B0h]
  __int64 v54; // [rsp+10h] [rbp-B0h]
  __int64 v55; // [rsp+10h] [rbp-B0h]
  __int64 v56; // [rsp+10h] [rbp-B0h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  __int64 v59; // [rsp+18h] [rbp-A8h]
  __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+18h] [rbp-A8h]
  _OWORD v62[3]; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v63; // [rsp+50h] [rbp-70h] BYREF
  __m128i v64; // [rsp+60h] [rbp-60h] BYREF
  __m128i v65; // [rsp+70h] [rbp-50h] BYREF
  char v66; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
    return 0;
  v6 = *a3;
  v7 = *(_DWORD *)(a2 + 12);
  v9 = a4;
  if ( v6 )
  {
    if ( v7 != *((unsigned __int16 *)a3 + 10) )
      return 0;
    if ( *((_BYTE *)a3 + 24) )
      return 0;
    v12 = a3[4];
    if ( v12 > 1 )
      return 0;
    if ( *((_BYTE *)a3 + 22) )
    {
      if ( !*(_BYTE *)(a2 + 16) && v12 )
        return 0;
      v13 = *((_QWORD *)a3 + 4);
    }
    else
    {
      v13 = *(_QWORD *)a2;
      v4 = *((_QWORD *)a3 + 4);
    }
    v10 = 0;
    if ( !*((_BYTE *)a3 + 23) )
      goto LABEL_11;
  }
  else
  {
    if ( v7 != -1 )
      return 0;
    v20 = *((_QWORD *)a3 + 4);
    v13 = v20;
    if ( *(_BYTE *)v20 == 61 )
    {
      v21 = *(_WORD *)(v20 + 2);
      if ( (v21 & 1) != 0 )
        return 0;
      if ( ((v21 >> 7) & 6) != 0 )
        return 0;
      if ( !*(_BYTE *)(a2 + 16) )
      {
        v47 = *(_QWORD *)a2;
        v52 = (unsigned __int8 *)*((_QWORD *)a3 + 4);
        v22 = sub_B46500(v52);
        v6 = 0;
        v13 = (__int64)v52;
        v4 = v47;
        v9 = a4;
        if ( v22 )
          return 0;
      }
    }
    else
    {
      if ( *(_BYTE *)v20 != 62 )
        return 0;
      v40 = *(_WORD *)(v20 + 2);
      if ( (v40 & 1) != 0 )
        return 0;
      v13 = *(_QWORD *)a2;
      v4 = *((_QWORD *)a3 + 4);
      if ( ((v40 >> 7) & 6) != 0 )
        return 0;
    }
    v10 = 0;
    if ( *(_BYTE *)v20 != 62 )
      goto LABEL_11;
  }
  v23 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 <= 0x1Cu )
    goto LABEL_35;
  if ( v23 != 85 )
  {
    if ( v23 == 61 )
    {
LABEL_55:
      v10 = v4;
      goto LABEL_36;
    }
LABEL_35:
    v10 = *(_QWORD *)(v4 - 64);
    goto LABEL_36;
  }
  v30 = *(_QWORD *)(v4 - 32);
  if ( !v30 || *(_BYTE *)v30 || *(_QWORD *)(v30 + 24) != *(_QWORD *)(v4 + 80) || (*(_BYTE *)(v30 + 33) & 0x20) == 0 )
    goto LABEL_35;
  v31 = *(_DWORD *)(v30 + 36);
  if ( v31 == 228 )
    goto LABEL_55;
  if ( v31 == 230 )
  {
    v10 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
LABEL_36:
    if ( *(_QWORD *)(v13 + 8) != *(_QWORD *)(v10 + 8) )
      v10 = 0;
    goto LABEL_38;
  }
  v50 = v9;
  v56 = v13;
  v61 = v4;
  v41 = sub_DFDDB0(*(_QWORD *)(a1 + 8));
  v6 = *a3;
  v4 = v61;
  v13 = v56;
  v9 = v50;
  v10 = v41;
LABEL_38:
  if ( v6 )
  {
    if ( !*((_BYTE *)a3 + 23) )
      goto LABEL_11;
  }
  else if ( **((_BYTE **)a3 + 4) != 62 )
  {
    goto LABEL_11;
  }
  if ( *(_QWORD *)a2 != v10 )
    return 0;
LABEL_11:
  v14 = *(_BYTE *)v13;
  if ( *(_BYTE *)v4 == 85
    && (v29 = *(_QWORD *)(v4 - 32)) != 0
    && !*(_BYTE *)v29
    && *(_QWORD *)(v29 + 24) == *(_QWORD *)(v4 + 80)
    && (*(_BYTE *)(v29 + 33) & 0x20) != 0
    && (*(_DWORD *)(v29 + 36) & 0xFFFFFFFD) == 0xE4 )
  {
    if ( v14 != 85 )
      return 0;
    v15 = *(_QWORD *)(v13 - 32);
    if ( !v15 || *(_BYTE *)v15 )
      return 0;
    v16 = 1;
  }
  else
  {
    if ( v14 != 85 )
      goto LABEL_18;
    v15 = *(_QWORD *)(v13 - 32);
    if ( !v15 || *(_BYTE *)v15 )
      goto LABEL_18;
    v16 = 0;
  }
  if ( *(_QWORD *)(v15 + 24) != *(_QWORD *)(v13 + 80)
    || (*(_BYTE *)(v15 + 33) & 0x20) == 0
    || (*(_DWORD *)(v15 + 36) & 0xFFFFFFFD) != 0xE4 )
  {
    if ( !v16 )
    {
LABEL_18:
      v17 = *((_QWORD *)a3 + 4);
      goto LABEL_19;
    }
    return 0;
  }
  if ( !v16 )
    return 0;
  v32 = *(_QWORD *)a2;
  v17 = *((_QWORD *)a3 + 4);
  v33 = *(_QWORD *)(*(_QWORD *)a2 - 32LL);
  if ( !v33 || *(_BYTE *)v33 || *(_QWORD *)(v33 + 24) != *(_QWORD *)(v32 + 80) )
    goto LABEL_108;
  v34 = *(_DWORD *)(v33 + 36);
  if ( v34 == 228 )
  {
    v35 = *(_QWORD *)(v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF));
  }
  else
  {
    if ( v34 != 230 )
      goto LABEL_108;
    v35 = *(_QWORD *)(v32 + 32 * (1LL - (*(_DWORD *)(v32 + 4) & 0x7FFFFFF)));
  }
  v36 = *(_QWORD *)(v17 - 32);
  if ( !v36 || *(_BYTE *)v36 || *(_QWORD *)(v36 + 24) != *(_QWORD *)(v17 + 80) )
    goto LABEL_108;
  v37 = *(_DWORD *)(v36 + 36);
  if ( v37 == 228 )
  {
    v38 = *(_QWORD *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
    goto LABEL_87;
  }
  if ( v37 != 230 )
LABEL_108:
    BUG();
  v38 = *(_QWORD *)(v17 + 32 * (1LL - (*(_DWORD *)(v17 + 4) & 0x7FFFFFF)));
LABEL_87:
  v49 = v9;
  v55 = v4;
  v60 = v13;
  if ( v38 != v35 )
    return 0;
  v39 = sub_277A9A0(v32, v17);
  v13 = v60;
  v4 = v55;
  v9 = v49;
  if ( !v39 )
    return 0;
LABEL_19:
  v44 = *(_DWORD *)(a2 + 8);
  if ( *(_BYTE *)v17 != 61
    || (*(_BYTE *)(v17 + 7) & 0x20) == 0
    || (v46 = v9, v51 = v4, v57 = v13, v18 = sub_B91C10(v17, 6), v13 = v57, v4 = v51, v9 = v46, !v18) )
  {
    v48 = v9;
    v53 = v4;
    v58 = v13;
    sub_D66840(&v63, (_BYTE *)v17);
    v13 = v58;
    v4 = v53;
    v24 = v48;
    if ( !v66 )
      goto LABEL_111;
    v25 = _mm_loadu_si128(&v64);
    v26 = _mm_loadu_si128(&v65);
    v62[0] = _mm_loadu_si128(&v63);
    v62[1] = v25;
    v62[2] = v26;
    v27 = sub_277D3C0(a1 + 512, (__int64 *)v62);
    v13 = v58;
    v4 = v53;
    v24 = v48;
    if ( !v27
      || v27 != (_QWORD *)(*(_QWORD *)(a1 + 520) + 56LL * *(unsigned int *)(a1 + 536))
      && v44 < *(_DWORD *)(v27[6] + 64LL) )
    {
LABEL_111:
      v54 = v4;
      v59 = v13;
      v28 = sub_277B370(a1, *(_DWORD *)(a2 + 8), v24, *(_QWORD *)a2, *((_QWORD *)a3 + 4));
      v13 = v59;
      v4 = v54;
      if ( !v28 )
        return 0;
    }
  }
  if ( v10 )
    return v10;
  v19 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 <= 0x1Cu )
    goto LABEL_24;
  if ( v19 != 85 )
  {
    if ( v19 == 61 )
    {
LABEL_99:
      v10 = v4;
      goto LABEL_25;
    }
LABEL_24:
    v10 = *(_QWORD *)(v4 - 64);
    goto LABEL_25;
  }
  v42 = *(_QWORD *)(v4 - 32);
  if ( !v42 || *(_BYTE *)v42 || *(_QWORD *)(v42 + 24) != *(_QWORD *)(v4 + 80) || (*(_BYTE *)(v42 + 33) & 0x20) == 0 )
    goto LABEL_24;
  v43 = *(_DWORD *)(v42 + 36);
  if ( v43 == 228 )
    goto LABEL_99;
  if ( v43 == 230 )
  {
    v10 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
LABEL_25:
    if ( *(_QWORD *)(v13 + 8) == *(_QWORD *)(v10 + 8) )
      return v10;
    return 0;
  }
  return sub_DFDDB0(*(_QWORD *)(a1 + 8));
}
