// Function: sub_1997F10
// Address: 0x1997f10
//
char __fastcall sub_1997F10(
        __int64 *a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // r9
  __int64 v16; // r12
  __int64 *v17; // rcx
  bool v18; // zf
  __int64 v19; // rax
  _BOOL4 v20; // esi
  __int64 *v21; // rcx
  unsigned int v22; // edi
  __int64 v23; // rax
  double v24; // xmm0_8
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 *v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // r10
  unsigned int v30; // r8d
  __int64 v31; // rdi
  __int64 v32; // r13
  __int64 *v33; // r15
  __int64 v34; // rt0
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // r10
  __int64 v38; // r14
  __int64 v39; // rax
  __m128 v40; // xmm0
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  double v51; // xmm4_8
  double v52; // xmm5_8
  __int64 v54; // [rsp+10h] [rbp-80h]
  __int64 v55; // [rsp+10h] [rbp-80h]
  __int64 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  int v58; // [rsp+18h] [rbp-78h]
  __int64 *v59; // [rsp+20h] [rbp-70h]
  _BOOL8 v60; // [rsp+20h] [rbp-70h]
  __int64 v61; // [rsp+28h] [rbp-68h]
  __int64 v62; // [rsp+28h] [rbp-68h]
  __int64 *v63; // [rsp+30h] [rbp-60h]
  int v64; // [rsp+30h] [rbp-60h]
  __int64 v65; // [rsp+30h] [rbp-60h]
  _BOOL8 v66; // [rsp+30h] [rbp-60h]
  char v67; // [rsp+38h] [rbp-58h]
  __int64 v68; // [rsp+38h] [rbp-58h]
  __int64 v69[2]; // [rsp+40h] [rbp-50h] BYREF
  char v70; // [rsp+50h] [rbp-40h]
  char v71; // [rsp+51h] [rbp-3Fh]

  v9 = sub_1481F60((_QWORD *)a1[1], a1[5], a2, a3);
  LOBYTE(v10) = sub_14562D0(v9);
  if ( (_BYTE)v10 )
    return v10;
  v12 = *(_QWORD *)(*a1 + 216);
  v13 = *a1 + 208;
  if ( v12 == v13 )
    return v10;
  while ( 1 )
  {
    v10 = v12;
    v12 = *(_QWORD *)(v12 + 8);
    v16 = *(_QWORD *)(v10 - 8);
    LOBYTE(v10) = *(_BYTE *)(v16 + 16);
    if ( (_BYTE)v10 == 65 )
    {
      v67 = 0;
      v14 = *(_QWORD *)v16;
    }
    else
    {
      if ( (_BYTE)v10 != 66 )
        goto LABEL_9;
      v67 = 1;
      v14 = *(_QWORD *)v16;
    }
    if ( !v14 )
      goto LABEL_9;
    LOBYTE(v10) = sub_14A2D80(a1[4]);
    if ( !(_BYTE)v10 )
      goto LABEL_9;
    if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
    {
      v10 = *(_QWORD *)(v16 - 8);
      v15 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 16LL) != 77 )
        goto LABEL_9;
    }
    else
    {
      v10 = v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
      v15 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 16LL) != 77 )
        goto LABEL_9;
    }
    LODWORD(v10) = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
    if ( (_DWORD)v10 != 2 )
      goto LABEL_9;
    v63 = (__int64 *)v15;
    v10 = sub_146F1B0(a1[1], v15);
    if ( *(_WORD *)(v10 + 24) != 7 )
      goto LABEL_9;
    LOWORD(v10) = *(_WORD *)(v10 + 26);
    if ( v67 )
    {
      if ( (v10 & 4) == 0 )
        goto LABEL_9;
    }
    else if ( (v10 & 2) == 0 )
    {
      goto LABEL_9;
    }
    v59 = v63;
    v61 = *v63;
    LODWORD(v10) = sub_16431F0(v14);
    v64 = v10;
    if ( (_DWORD)v10 == -1 )
      goto LABEL_9;
    LODWORD(v10) = sub_1456C90(a1[1], v61);
    if ( v64 < (int)v10 )
      goto LABEL_9;
    if ( (*((_BYTE *)v59 + 23) & 0x40) != 0 )
      v17 = (__int64 *)*(v59 - 1);
    else
      v17 = &v59[-3 * (*((_DWORD *)v59 + 5) & 0xFFFFFFF)];
    v65 = v17[3 * *((unsigned int *)v59 + 14) + 1];
    v18 = sub_13FC520(a1[5]) == v65;
    v19 = 3;
    if ( v18 )
      v19 = 0;
    v20 = v18;
    v66 = !v18;
    v21 = (*((_BYTE *)v59 + 23) & 0x40) != 0 ? (__int64 *)*(v59 - 1) : &v59[-3 * (*((_DWORD *)v59 + 5) & 0xFFFFFFF)];
    v10 = v21[v19];
    if ( *(_BYTE *)(v10 + 16) != 13 )
      goto LABEL_9;
    v22 = *(_DWORD *)(v10 + 32);
    v23 = *(_QWORD *)(v10 + 24);
    if ( v67 )
    {
      if ( v22 > 0x40 )
        v23 = *(_QWORD *)v23;
      else
        v23 = v23 << (64 - (unsigned __int8)v22) >> (64 - (unsigned __int8)v22);
LABEL_30:
      v24 = (double)(int)v23;
      goto LABEL_31;
    }
    if ( v22 > 0x40 )
      v23 = *(_QWORD *)v23;
    if ( v23 >= 0 )
      goto LABEL_30;
    v24 = (double)(int)(v23 & 1 | ((unsigned __int64)v23 >> 1)) + (double)(int)(v23 & 1 | ((unsigned __int64)v23 >> 1));
LABEL_31:
    v25 = sub_15A10B0(v14, v24);
    v26 = (__int64)v59;
    v62 = v25;
    if ( (*((_BYTE *)v59 + 23) & 0x40) != 0 )
      v27 = (__int64 *)*(v59 - 1);
    else
      v27 = &v59[-3 * (*((_DWORD *)v59 + 5) & 0xFFFFFFF)];
    v60 = v20;
    v28 = v27[3 * v20];
    LOBYTE(v10) = *(_BYTE *)(v28 + 16) - 35;
    if ( (unsigned __int8)v10 > 0x11u || (v10 & 0xFD) != 0 )
      goto LABEL_9;
    v29 = *(_QWORD *)(v28 - 48);
    v10 = *(_QWORD *)(v28 - 24);
    if ( v29 && v26 == v29 )
    {
      v29 = *(_QWORD *)(v28 - 24);
      if ( *(_BYTE *)(v10 + 16) != 13 )
        goto LABEL_9;
    }
    else if ( !v10 || v10 != v26 || *(_BYTE *)(v29 + 16) != 13 )
    {
      goto LABEL_9;
    }
    v30 = *(_DWORD *)(v29 + 32);
    v31 = *(_QWORD *)(v29 + 24);
    v10 = 1LL << ((unsigned __int8)v30 - 1);
    if ( v30 > 0x40 )
      break;
    if ( (v10 & v31) == 0 && v31 )
      goto LABEL_43;
LABEL_9:
    if ( v12 == v13 )
      return v10;
  }
  v10 &= *(_QWORD *)(v31 + 8LL * ((v30 - 1) >> 6));
  if ( v10 )
    goto LABEL_9;
  v58 = *(_DWORD *)(v29 + 32);
  v55 = v26;
  v68 = v29;
  LODWORD(v10) = sub_16A57B0(v29 + 24);
  if ( v58 == (_DWORD)v10 )
    goto LABEL_9;
  v26 = v55;
  v29 = v68;
LABEL_43:
  v54 = v29;
  v34 = v14;
  v33 = a1;
  v32 = v34;
  v56 = v26;
  v71 = 1;
  v69[0] = (__int64)"IV.S.";
  v70 = 3;
  v35 = sub_1648B60(64);
  v36 = v56;
  v37 = v54;
  v38 = v35;
  if ( v35 )
  {
    sub_15F1EA0(v35, v32, 53, 0, 0, v56);
    *(_DWORD *)(v38 + 56) = 2;
    sub_164B780(v38, v69);
    sub_1648880(v38, *(_DWORD *)(v38 + 56), 1);
    v37 = v54;
    v36 = v56;
  }
  v39 = *(_QWORD *)(v37 + 24);
  if ( *(_DWORD *)(v37 + 32) > 0x40u )
    v39 = *(_QWORD *)v39;
  v40 = 0;
  if ( v39 < 0 )
    *(double *)v40.m128_u64 = (double)(int)(v39 & 1 | ((unsigned __int64)v39 >> 1))
                            + (double)(int)(v39 & 1 | ((unsigned __int64)v39 >> 1));
  else
    *(double *)v40.m128_u64 = (double)(int)v39;
  v57 = v36;
  v41 = sub_15A10B0(v32, *(double *)v40.m128_u64);
  v71 = 1;
  v69[0] = (__int64)"IV.S.next.";
  v70 = 3;
  v42 = sub_15FB440(2 * (unsigned int)(*(_BYTE *)(v28 + 16) != 35) + 12, (__int64 *)v38, v41, (__int64)v69, v28);
  v43 = sub_193FF80(v57);
  sub_1704F80(v38, v62, *(_QWORD *)(v43 + 8 * v66), v44, v45, v46);
  v47 = sub_193FF80(v57);
  sub_1704F80(v38, v42, *(_QWORD *)(v47 + 8 * v60), v48, v49, v50);
  sub_164D160(v16, v38, v40, *(double *)a3.m128i_i64, a4, a5, v51, v52, a8, a9);
  LOBYTE(v10) = sub_15F20C0((_QWORD *)v16);
  *((_BYTE *)v33 + 48) = 1;
  return v10;
}
