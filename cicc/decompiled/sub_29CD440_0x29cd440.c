// Function: sub_29CD440
// Address: 0x29cd440
//
void __fastcall sub_29CD440(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        char a6,
        __int64 *a7)
{
  __int64 v8; // rdi
  __int64 v11; // r12
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // r15
  _BYTE *v16; // r8
  size_t v17; // rbx
  _QWORD *v18; // rax
  unsigned __int64 v19; // rax
  int v20; // edx
  __int64 *v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rcx
  _QWORD *v27; // rax
  __int64 v28; // r9
  __int64 v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rdi
  __int64 v37; // r15
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __m128i *v41; // r13
  unsigned __int16 v42; // ax
  __m128i *v43; // r15
  __int64 v44; // rsi
  _QWORD *v45; // rax
  __int64 v46; // rbx
  __int64 v47; // rdi
  unsigned __int16 v48; // ax
  __m128i *v49; // r13
  __int64 v50; // rsi
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD **v54; // r11
  __m128i *v55; // rax
  __m128i *v56; // rbx
  unsigned __int16 v57; // dx
  __m128i *v58; // r8
  __int64 v59; // rsi
  __int64 *v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // r15
  __int64 v64; // rdx
  __int64 v65; // r12
  _QWORD *v66; // rax
  __int64 v67; // rdi
  unsigned __int16 v68; // ax
  __m128i *v69; // r12
  __int64 v70; // rsi
  __int64 v71; // rsi
  unsigned __int8 *v72; // rsi
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  _QWORD *v75; // rbx
  _QWORD *v76; // rax
  __int64 v77; // r9
  __int64 *v78; // rax
  unsigned __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rax
  _QWORD *v82; // rdi
  __int64 v83; // rsi
  unsigned __int8 *v84; // rsi
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // [rsp+8h] [rbp-178h]
  _QWORD **src; // [rsp+18h] [rbp-168h]
  __m128i *srcb; // [rsp+18h] [rbp-168h]
  void *srca; // [rsp+18h] [rbp-168h]
  _QWORD *srcc; // [rsp+18h] [rbp-168h]
  _BYTE *srcd; // [rsp+18h] [rbp-168h]
  __int64 v93; // [rsp+20h] [rbp-160h]
  __int64 v95; // [rsp+28h] [rbp-158h]
  __int64 v96; // [rsp+28h] [rbp-158h]
  __int64 v100; // [rsp+40h] [rbp-140h]
  __int64 v101; // [rsp+48h] [rbp-138h]
  __m128i v102[2]; // [rsp+50h] [rbp-130h] BYREF
  char v103; // [rsp+70h] [rbp-110h]
  char v104; // [rsp+71h] [rbp-10Fh]
  __m128i v105[2]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v106; // [rsp+A0h] [rbp-E0h]
  __m128i v107[3]; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v108; // [rsp+E0h] [rbp-A0h] BYREF
  _BYTE v109[16]; // [rsp+F0h] [rbp-90h] BYREF
  __int16 v110; // [rsp+100h] [rbp-80h]
  __m128i v111; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v112[2]; // [rsp+120h] [rbp-60h] BYREF
  __int64 v113; // [rsp+130h] [rbp-50h]
  __int64 v114; // [rsp+138h] [rbp-48h]
  __int64 v115; // [rsp+140h] [rbp-40h]

  if ( !a4 )
    BUG();
  v8 = *(_QWORD *)(a4 + 16);
  v11 = *(_QWORD *)(*(_QWORD *)(v8 + 72) + 40LL);
  v15 = (__int64 *)sub_AA48A0(v8);
  if ( a3 == 6 )
  {
    if ( *(_DWORD *)a2 == 1970234221 && *(_WORD *)(a2 + 4) == 29806 )
      goto LABEL_9;
    goto LABEL_21;
  }
  if ( a3 == 7 )
  {
    if ( *(_DWORD *)a2 == 1868786990 && *(_WORD *)(a2 + 4) == 28277 && *(_BYTE *)(a2 + 6) == 116 )
      goto LABEL_9;
LABEL_5:
    if ( *(_DWORD *)a2 == 1868786945 && *(_WORD *)(a2 + 4) == 28277 && *(_BYTE *)(a2 + 6) == 116
      || *(_DWORD *)a2 == 1868787039 && *(_WORD *)(a2 + 4) == 28277 && *(_BYTE *)(a2 + 6) == 116 )
    {
LABEL_9:
      v111.m128i_i64[0] = (__int64)v112;
      v16 = *(_BYTE **)(v11 + 232);
      v17 = *(_QWORD *)(v11 + 240);
      if ( &v16[v17] && !v16 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v108.m128i_i64[0] = *(_QWORD *)(v11 + 240);
      if ( v17 > 0xF )
      {
        srcd = v16;
        v81 = sub_22409D0((__int64)&v111, (unsigned __int64 *)&v108, 0);
        v16 = srcd;
        v111.m128i_i64[0] = v81;
        v82 = (_QWORD *)v81;
        v112[0] = v108.m128i_i64[0];
      }
      else
      {
        if ( v17 == 1 )
        {
          LOBYTE(v112[0]) = *v16;
          v18 = v112;
LABEL_14:
          v111.m128i_i64[1] = v17;
          *((_BYTE *)v18 + v17) = 0;
          v19 = *(unsigned int *)(v11 + 264);
          v20 = *(_DWORD *)(v11 + 276);
          v113 = *(_QWORD *)(v11 + 264);
          v114 = *(_QWORD *)(v11 + 272);
          v115 = *(_QWORD *)(v11 + 280);
          if ( v20 == 19 && a3 == 8 && *(_QWORD *)a2 == 0x746E756F636D5F5FLL )
          {
            v75 = (_QWORD *)sub_AE4420(v11 + 312, (__int64)v15, 0);
            v96 = sub_BCE3C0(v15, 0);
            v107[0].m128i_i8[4] = 0;
            srca = (void *)sub_AD64C0((__int64)v75, 0, 0);
            v110 = 257;
            v76 = sub_BD2C40(88, unk_3F0FAE8);
            if ( v76 )
            {
              v77 = (__int64)srca;
              srcc = v76;
              sub_B30000((__int64)v76, v11, v75, 0, 7, v77, (__int64)&v108, 0, 0, v107[0].m128i_i64[0], 0);
              v76 = srcc;
            }
            v107[0].m128i_i64[0] = (__int64)v76;
            v110 = 257;
            v105[0].m128i_i64[0] = v96;
            v78 = (__int64 *)sub_BCB120(v15);
            v79 = sub_BCF480(v78, v105, 1, 0);
            v63 = sub_BA8CA0(v11, a2, 8u, v79);
            v65 = v80;
            goto LABEL_57;
          }
          if ( (unsigned int)v19 > 0x1D || (v51 = 805331000, !_bittest64(&v51, v19)) )
          {
            v21 = (__int64 *)sub_BCB120(v15);
            v108.m128i_i64[0] = (__int64)v109;
            v108.m128i_i64[1] = 0;
            v22 = sub_BCF480(v21, v109, 0, 0);
            v23 = sub_BA8C10(v11, a2, a3, v22, 0);
            v25 = v24;
            if ( (_BYTE *)v108.m128i_i64[0] != v109 )
              _libc_free(v108.m128i_u64[0]);
            v26 = a5;
            v110 = 257;
            BYTE1(v26) = a6;
            v101 = v26;
            v27 = sub_BD2C40(88, 1u);
            v29 = (__int64)v27;
            if ( v27 )
              sub_B4A410((__int64)v27, v23, v25, (__int64)&v108, 1u, v28, a4, v101);
LABEL_59:
            v69 = (__m128i *)(v29 + 48);
            v70 = *a7;
            v108.m128i_i64[0] = v70;
            if ( v70 )
            {
              sub_B96E90((__int64)&v108, v70, 1);
              if ( v69 == &v108 )
              {
                if ( v108.m128i_i64[0] )
                  sub_B91220((__int64)&v108, v108.m128i_i64[0]);
                goto LABEL_63;
              }
              v71 = *(_QWORD *)(v29 + 48);
              if ( !v71 )
              {
LABEL_69:
                v72 = (unsigned __int8 *)v108.m128i_i64[0];
                *(_QWORD *)(v29 + 48) = v108.m128i_i64[0];
                if ( v72 )
                  sub_B976B0((__int64)&v108, v72, v29 + 48);
                goto LABEL_63;
              }
            }
            else if ( v69 == &v108 || (v71 = *(_QWORD *)(v29 + 48)) == 0 )
            {
LABEL_63:
              if ( (_QWORD *)v111.m128i_i64[0] != v112 )
                j_j___libc_free_0(v111.m128i_u64[0]);
              return;
            }
            sub_B91220(v29 + 48, v71);
            goto LABEL_69;
          }
          v110 = 257;
          v52 = sub_BCB2D0(v15);
          v107[0].m128i_i64[0] = sub_ACD640(v52, 0, 0);
          v53 = sub_B6E160((__int64 *)v11, 0x133u, 0, 0);
          v54 = 0;
          if ( v53 )
            v54 = *(_QWORD ***)(v53 + 24);
          src = v54;
          v87 = v53;
          v55 = (__m128i *)sub_BD2C40(88, 2u);
          v56 = v55;
          if ( v55 )
          {
            LOBYTE(v57) = a5;
            HIBYTE(v57) = a6;
            sub_B44260((__int64)v55, *src[2], 56, 2u, a4, v57);
            v56[4].m128i_i64[1] = 0;
            sub_B4A290((__int64)v56, (__int64)src, v87, v107[0].m128i_i64, 1, (__int64)&v108, 0, 0);
          }
          v58 = v56 + 3;
          v59 = *a7;
          v108.m128i_i64[0] = v59;
          if ( v59 )
          {
            sub_B96E90((__int64)&v108, v59, 1);
            v58 = v56 + 3;
            if ( &v56[3] == &v108 )
            {
              if ( v108.m128i_i64[0] )
                sub_B91220((__int64)&v108, v108.m128i_i64[0]);
              goto LABEL_56;
            }
            v73 = v56[3].m128i_i64[0];
            if ( !v73 )
            {
LABEL_74:
              v74 = (unsigned __int8 *)v108.m128i_i64[0];
              v56[3].m128i_i64[0] = v108.m128i_i64[0];
              if ( v74 )
                sub_B976B0((__int64)&v108, v74, (__int64)v58);
              goto LABEL_56;
            }
          }
          else if ( v58 == &v108 || (v73 = v56[3].m128i_i64[0]) == 0 )
          {
LABEL_56:
            v108.m128i_i64[0] = sub_BCE3C0(v15, 0);
            v60 = (__int64 *)sub_BCB120(v15);
            v61 = sub_BCF480(v60, &v108, 1, 0);
            v62 = sub_BA8CA0(v11, a2, a3, v61);
            v107[0].m128i_i64[0] = (__int64)v56;
            v110 = 257;
            v63 = v62;
            v65 = v64;
LABEL_57:
            v66 = sub_BD2C40(88, 2u);
            v29 = (__int64)v66;
            if ( v66 )
            {
              v67 = (__int64)v66;
              LOBYTE(v68) = a5;
              HIBYTE(v68) = a6;
              sub_B44260(v67, **(_QWORD **)(v63 + 16), 56, 2u, a4, v68);
              *(_QWORD *)(v29 + 72) = 0;
              sub_B4A290(v29, v63, v65, v107[0].m128i_i64, 1, (__int64)&v108, 0, 0);
            }
            goto LABEL_59;
          }
          srcb = v58;
          sub_B91220((__int64)v58, v73);
          v58 = srcb;
          goto LABEL_74;
        }
        if ( !v17 )
        {
          v18 = v112;
          goto LABEL_14;
        }
        v82 = v112;
      }
      memcpy(v82, v16, v17);
      v17 = v108.m128i_i64[0];
      v18 = (_QWORD *)v111.m128i_i64[0];
      goto LABEL_14;
    }
LABEL_21:
    v108.m128i_i64[0] = (__int64)"'";
    v110 = 259;
    v105[0].m128i_i64[0] = a2;
    v106 = 261;
    v105[0].m128i_i64[1] = a3;
    v102[0].m128i_i64[0] = (__int64)"Unknown instrumentation function: '";
    v104 = 1;
    v103 = 3;
    sub_9C6370(v107, v102, v105, v12, v13, v14);
    sub_9C6370(&v111, v107, &v108, v30, v31, v32);
    sub_C64D30((__int64)&v111, 1u);
  }
  switch ( a3 )
  {
    case 0x18uLL:
      if ( !(*(_QWORD *)a2 ^ 0x6D72612E6D766C6CLL | *(_QWORD *)(a2 + 8) ^ 0x6261652E756E672ELL)
        && *(_QWORD *)(a2 + 16) == 0x746E756F636D2E69LL )
      {
        goto LABEL_9;
      }
      v12 = a2;
      if ( *(_QWORD *)a2 ^ 0x72705F6779635F5FLL | *(_QWORD *)(a2 + 8) ^ 0x75665F656C69666FLL
        || *(_QWORD *)(a2 + 16) != 0x7265746E655F636ELL )
      {
        goto LABEL_21;
      }
      break;
    case 8uLL:
      if ( *(_QWORD *)a2 == 0x746E756F636D5F01LL )
        goto LABEL_9;
      v12 = a2;
      if ( *(_QWORD *)a2 == 0x746E756F636D5F5FLL )
        goto LABEL_9;
      goto LABEL_21;
    case 7uLL:
      goto LABEL_5;
    case 0x1DuLL:
      v12 = a2;
      if ( !(*(_QWORD *)a2 ^ 0x72705F6779635F5FLL | *(_QWORD *)(a2 + 8) ^ 0x75665F656C69666FLL)
        && *(_QWORD *)(a2 + 16) == 0x7265746E655F636ELL
        && *(_DWORD *)(a2 + 24) == 1918984799
        && *(_BYTE *)(a2 + 28) == 101 )
      {
        goto LABEL_9;
      }
      goto LABEL_21;
    case 0x17uLL:
      v12 = a2;
      if ( *(_QWORD *)a2 ^ 0x72705F6779635F5FLL | *(_QWORD *)(a2 + 8) ^ 0x75665F656C69666FLL
        || *(_DWORD *)(a2 + 16) != 1700750190
        || *(_WORD *)(a2 + 20) != 27000
        || *(_BYTE *)(a2 + 22) != 116 )
      {
        goto LABEL_21;
      }
      break;
    default:
      goto LABEL_21;
  }
  v107[0].m128i_i64[0] = sub_BCE3C0(v15, 0);
  v107[0].m128i_i64[1] = sub_BCE3C0(v15, 0);
  v33 = (__int64 *)sub_BCB120(v15);
  v34 = sub_BCF480(v33, v107, 2, 0);
  v35 = sub_BA8CA0(v11, a2, a3, v34);
  v36 = v15;
  v37 = 0;
  LOWORD(v113) = 257;
  v95 = v38;
  v100 = v35;
  v39 = sub_BCB2D0(v36);
  v108.m128i_i64[0] = sub_ACD640(v39, 0, 0);
  v40 = sub_B6E160((__int64 *)v11, 0x133u, 0, 0);
  if ( v40 )
    v37 = *(_QWORD *)(v40 + 24);
  v93 = v40;
  v41 = (__m128i *)sub_BD2C40(88, 2u);
  if ( v41 )
  {
    LOBYTE(v42) = a5;
    HIBYTE(v42) = a6;
    sub_B44260((__int64)v41, **(_QWORD **)(v37 + 16), 56, 2u, a4, v42);
    v41[4].m128i_i64[1] = 0;
    sub_B4A290((__int64)v41, v37, v93, v108.m128i_i64, 1, (__int64)&v111, 0, 0);
  }
  v43 = v41 + 3;
  v44 = *a7;
  v111.m128i_i64[0] = v44;
  if ( !v44 )
  {
    if ( v43 == &v111 )
      goto LABEL_34;
    v83 = v41[3].m128i_i64[0];
    if ( !v83 )
      goto LABEL_34;
LABEL_98:
    sub_B91220((__int64)v41[3].m128i_i64, v83);
    goto LABEL_99;
  }
  sub_B96E90((__int64)&v111, v44, 1);
  if ( v43 == &v111 )
  {
    if ( v111.m128i_i64[0] )
      sub_B91220((__int64)&v111, v111.m128i_i64[0]);
    goto LABEL_34;
  }
  v83 = v41[3].m128i_i64[0];
  if ( v83 )
    goto LABEL_98;
LABEL_99:
  v84 = (unsigned __int8 *)v111.m128i_i64[0];
  v41[3].m128i_i64[0] = v111.m128i_i64[0];
  if ( v84 )
    sub_B976B0((__int64)&v111, v84, (__int64)v41[3].m128i_i64);
LABEL_34:
  v108.m128i_i64[0] = a1;
  v108.m128i_i64[1] = (__int64)v41;
  LOWORD(v113) = 257;
  v45 = sub_BD2C40(88, 3u);
  v46 = (__int64)v45;
  if ( v45 )
  {
    v47 = (__int64)v45;
    LOBYTE(v48) = a5;
    HIBYTE(v48) = a6;
    sub_B44260(v47, **(_QWORD **)(v100 + 16), 56, 3u, a4, v48);
    *(_QWORD *)(v46 + 72) = 0;
    sub_B4A290(v46, v100, v95, v108.m128i_i64, 2, (__int64)&v111, 0, 0);
  }
  v49 = (__m128i *)(v46 + 48);
  v50 = *a7;
  v111.m128i_i64[0] = v50;
  if ( !v50 )
  {
    if ( v49 == &v111 )
      return;
    v85 = *(_QWORD *)(v46 + 48);
    if ( !v85 )
      return;
LABEL_103:
    sub_B91220(v46 + 48, v85);
    goto LABEL_104;
  }
  sub_B96E90((__int64)&v111, v50, 1);
  if ( v49 == &v111 )
  {
    if ( v111.m128i_i64[0] )
      sub_B91220((__int64)&v111, v111.m128i_i64[0]);
    return;
  }
  v85 = *(_QWORD *)(v46 + 48);
  if ( v85 )
    goto LABEL_103;
LABEL_104:
  v86 = (unsigned __int8 *)v111.m128i_i64[0];
  *(_QWORD *)(v46 + 48) = v111.m128i_i64[0];
  if ( v86 )
    sub_B976B0((__int64)&v111, v86, v46 + 48);
}
