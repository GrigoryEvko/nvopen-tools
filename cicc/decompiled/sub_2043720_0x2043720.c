// Function: sub_2043720
// Address: 0x2043720
//
__int64 __fastcall sub_2043720(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v6; // rdi
  __int64 *v7; // rax
  unsigned __int64 v8; // rbx
  __int64 (__fastcall *v9)(__int64, __int64); // rcx
  __int64 v10; // r9
  __int64 v11; // r15
  int v12; // eax
  __int64 v13; // r10
  __int16 v14; // dx
  bool v15; // cc
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // esi
  __int16 v19; // cx
  int v20; // r8d
  char v21; // di
  _DWORD *v22; // rax
  int v23; // esi
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // rax
  unsigned int v29; // r8d
  __int64 *v30; // rax
  __int64 v31; // rax
  char v32; // cl
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int *v35; // rdi
  __int64 *v36; // rdx
  __int64 v37; // r11
  int v38; // esi
  __int64 v39; // rcx
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 *v46; // r10
  __int64 v47; // r10
  __int64 v48; // rax
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // edx
  __int64 *v55; // r10
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 *v58; // rdx
  __int64 v59; // rdx
  __int64 *v60; // rsi
  __int64 v61; // r11
  __int64 v62; // rdx
  unsigned int v63; // edi
  __int64 *v64; // rdx
  __int64 v65; // rdx
  __int64 *v66; // rdx
  __int64 v67; // rax
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rax
  unsigned int v71; // [rsp+Ch] [rbp-44h]
  __int64 v72; // [rsp+10h] [rbp-40h]
  __int64 v73; // [rsp+18h] [rbp-38h]

  v3 = 80;
  v6 = *(_QWORD *)(a3 + 16);
  if ( *(_WORD *)(a2 + 24) != 186 )
    v3 = 40;
  v7 = (__int64 *)(*(_QWORD *)(a2 + 32) + v3);
  v8 = v7[1];
  v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 1096LL);
  if ( v9 == sub_2043530 )
  {
    v10 = *((unsigned int *)v7 + 2);
    v11 = *v7;
  }
  else
  {
    v51 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v9)(v6, *v7, v7[1]);
    v8 = v52;
    v11 = v51;
    v10 = (unsigned int)v52;
  }
  v12 = (*(_WORD *)(a2 + 26) >> 7) & 7;
  if ( v12 == 1 )
  {
    v48 = 80;
    if ( *(_WORD *)(a2 + 24) != 185 )
      v48 = 120;
    v49 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + v48);
    v50 = *(unsigned __int16 *)(v49 + 24);
    if ( v50 == 10 || v50 == 32 )
    {
      v53 = *(_QWORD *)(v49 + 88);
      v54 = *(_DWORD *)(v53 + 32);
      v55 = *(__int64 **)(v53 + 24);
      if ( v54 > 0x40 )
        v13 = *v55;
      else
        v13 = (__int64)((_QWORD)v55 << (64 - (unsigned __int8)v54)) >> (64 - (unsigned __int8)v54);
      goto LABEL_7;
    }
    goto LABEL_49;
  }
  v13 = 0;
  if ( v12 != 2 )
    goto LABEL_7;
  v41 = 80;
  if ( *(_WORD *)(a2 + 24) != 185 )
    v41 = 120;
  v42 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + v41);
  v43 = *(unsigned __int16 *)(v42 + 24);
  if ( v43 != 10 && v43 != 32 )
  {
LABEL_49:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_BYTE *)(a1 + 40) = 0;
    return a1;
  }
  v44 = *(_QWORD *)(v42 + 88);
  v45 = *(_DWORD *)(v44 + 32);
  v46 = *(__int64 **)(v44 + 24);
  if ( v45 > 0x40 )
    v47 = *v46;
  else
    v47 = (__int64)((_QWORD)v46 << (64 - (unsigned __int8)v45)) >> (64 - (unsigned __int8)v45);
  v13 = -v47;
LABEL_7:
  while ( 1 )
  {
    v14 = *(_WORD *)(v11 + 24);
    v15 = v14 <= 119;
    if ( v14 == 119 )
      break;
    while ( !v15 )
    {
      if ( (unsigned __int16)(v14 - 185) > 1u )
        goto LABEL_33;
      v23 = (*(_WORD *)(v11 + 26) >> 7) & 7;
      v24 = (v14 == 185) == (_DWORD)v10 && v23 != 0;
      if ( !v24 )
        goto LABEL_44;
      v25 = 120;
      v26 = *(_QWORD *)(v11 + 32);
      if ( v14 == 185 )
        v25 = 80;
      v27 = *(_QWORD *)(v26 + v25);
      v24 = *(_WORD *)(v27 + 24) == 32 || *(_WORD *)(v27 + 24) == 10;
      if ( !v24 )
        goto LABEL_44;
      v28 = *(_QWORD *)(v27 + 88);
      v29 = *(_DWORD *)(v28 + 32);
      v30 = *(__int64 **)(v28 + 24);
      if ( v29 > 0x40 )
        v31 = *v30;
      else
        v31 = (__int64)((_QWORD)v30 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
      v32 = v23 - 2;
      v33 = v13 + v31;
      v13 -= v31;
      v34 = 80;
      if ( (v32 & 0xFD) != 0 )
        v13 = v33;
      if ( v14 != 186 )
        v34 = 40;
      v35 = (unsigned int *)(v34 + v26);
      v11 = *(_QWORD *)v35;
      v10 = v35[2];
      v14 = *(_WORD *)(*(_QWORD *)v35 + 24LL);
      v8 = v10 | v8 & 0xFFFFFFFF00000000LL;
      v15 = v14 <= 119;
      if ( v14 == 119 )
        goto LABEL_29;
    }
    if ( v14 != 52 )
    {
LABEL_33:
      v38 = 0;
      v39 = 0;
      goto LABEL_34;
    }
    v16 = *(_QWORD *)(v11 + 32);
    v17 = *(_QWORD *)(v16 + 40);
    v18 = *(unsigned __int16 *)(v17 + 24);
    v19 = *(_WORD *)(v17 + 24);
    if ( (_WORD)v18 != 10 && v18 != 32 )
      goto LABEL_12;
    v56 = *(_QWORD *)(v17 + 88);
    v57 = *(_DWORD *)(v56 + 32);
    v58 = *(__int64 **)(v56 + 24);
    if ( v57 > 0x40 )
      v59 = *v58;
    else
      v59 = (__int64)((_QWORD)v58 << (64 - (unsigned __int8)v57)) >> (64 - (unsigned __int8)v57);
    v13 += v59;
LABEL_56:
    v11 = *(_QWORD *)v16;
    v10 = *(unsigned int *)(v16 + 8);
    v8 = v10 | v8 & 0xFFFFFFFF00000000LL;
  }
LABEL_29:
  v36 = *(__int64 **)(v11 + 32);
  v37 = v36[5];
  v24 = *(_WORD *)(v37 + 24) == 10 || *(_WORD *)(v37 + 24) == 32;
  if ( !v24 )
    goto LABEL_44;
  v72 = v13;
  v73 = v36[5];
  v71 = v10;
  v24 = sub_1D1F940(a3, *v36, v36[1], *(_QWORD *)(v37 + 88) + 24LL, 0);
  v13 = v72;
  if ( v24 )
  {
    v67 = *(_QWORD *)(v73 + 88);
    v68 = *(_DWORD *)(v67 + 32);
    v69 = *(__int64 **)(v67 + 24);
    if ( v68 <= 0x40 )
      v70 = (__int64)((_QWORD)v69 << (64 - (unsigned __int8)v68)) >> (64 - (unsigned __int8)v68);
    else
      v70 = *v69;
    v13 = v70 + v72;
    v16 = *(_QWORD *)(v11 + 32);
    goto LABEL_56;
  }
  v10 = v71;
  if ( *(_WORD *)(v11 + 24) != 52 )
  {
LABEL_44:
    v38 = 0;
    v39 = 0;
LABEL_35:
    *(_QWORD *)a1 = v11;
    *(_QWORD *)(a1 + 16) = v39;
    *(_DWORD *)(a1 + 24) = v38;
    *(_QWORD *)(a1 + 8) = v10 | v8 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)(a1 + 32) = v13;
    *(_BYTE *)(a1 + 40) = v24;
    return a1;
  }
  v16 = *(_QWORD *)(v11 + 32);
  v17 = *(_QWORD *)(v16 + 40);
  v18 = *(unsigned __int16 *)(v17 + 24);
  v19 = *(_WORD *)(v17 + 24);
LABEL_12:
  if ( v18 != 54 )
  {
    v20 = *(_DWORD *)(v16 + 48);
    v11 = *(_QWORD *)v16;
    v21 = 0;
    v10 = *(unsigned int *)(v16 + 8);
    if ( v18 == 142 )
    {
      v22 = *(_DWORD **)(v17 + 32);
      v21 = 1;
      v17 = *(_QWORD *)v22;
      v20 = v22[2];
      v19 = *(_WORD *)(*(_QWORD *)v22 + 24LL);
    }
    if ( v19 != 52
      || (v60 = *(__int64 **)(v17 + 32),
          v61 = v60[5],
          (v24 = *(_WORD *)(v61 + 24) == 32 || *(_WORD *)(v61 + 24) == 10) == 0) )
    {
      *(_QWORD *)a1 = v11;
      *(_DWORD *)(a1 + 8) = v10;
      *(_QWORD *)(a1 + 16) = v17;
      *(_DWORD *)(a1 + 24) = v20;
      *(_QWORD *)(a1 + 32) = v13;
      *(_BYTE *)(a1 + 40) = v21;
      return a1;
    }
    v62 = *(_QWORD *)(v61 + 88);
    v63 = *(_DWORD *)(v62 + 32);
    v64 = *(__int64 **)(v62 + 24);
    if ( v63 > 0x40 )
      v65 = *v64;
    else
      v65 = (__int64)((_QWORD)v64 << (64 - (unsigned __int8)v63)) >> (64 - (unsigned __int8)v63);
    v39 = *v60;
    v13 += v65;
    v38 = *((_DWORD *)v60 + 2);
    if ( *(_WORD *)(v39 + 24) == 142 )
    {
      v66 = *(__int64 **)(v39 + 32);
      v39 = *v66;
      v38 = *((_DWORD *)v66 + 2);
    }
    else
    {
LABEL_34:
      v24 = 0;
    }
    goto LABEL_35;
  }
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 32) = v13;
  *(_BYTE *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 8) = v10 | v8 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  return a1;
}
