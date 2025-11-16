// Function: sub_3273860
// Address: 0x3273860
//
__int64 __fastcall sub_3273860(int a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v6; // r13d
  unsigned int v7; // r12d
  int v9; // edi
  unsigned int v10; // r14d
  __int64 v11; // r15
  __int64 v12; // rdx
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // edx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  char v22; // r14
  __int64 v23; // r13
  int v25; // ebx
  bool v26; // zf
  __int64 v27; // rbx
  unsigned __int16 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  char v31; // al
  _BYTE *v32; // rax
  unsigned __int16 v33; // r10
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rax
  char v37; // cl
  unsigned __int64 v38; // rsi
  int v39; // edx
  __int64 v40; // rbx
  __int64 v41; // rsi
  _QWORD *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int16 v45; // ax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rsi
  __int64 v51; // rax
  unsigned int v52; // edx
  int v53; // r13d
  __int64 v54; // rsi
  int v55; // edx
  __int64 v56; // rax
  char v57; // r12
  unsigned __int64 v58; // rcx
  int v59; // edx
  unsigned __int64 v60; // rbx
  __int64 v61; // rsi
  __int64 v62; // rsi
  __int128 v63; // rax
  __int64 v64; // rax
  unsigned int v65; // edx
  unsigned int v66; // ebx
  unsigned __int16 v67; // r10
  _BYTE *v68; // rax
  __int64 v69; // rbx
  char v70; // si
  __int64 v71; // rbx
  char v72; // di
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int128 v75; // [rsp-10h] [rbp-180h]
  __int64 v76; // [rsp+8h] [rbp-168h]
  unsigned __int16 v77; // [rsp+10h] [rbp-160h]
  unsigned __int16 v78; // [rsp+10h] [rbp-160h]
  unsigned int v79; // [rsp+28h] [rbp-148h]
  __int64 v80; // [rsp+30h] [rbp-140h]
  unsigned __int16 v81; // [rsp+30h] [rbp-140h]
  unsigned __int16 v82; // [rsp+30h] [rbp-140h]
  unsigned __int16 v83; // [rsp+30h] [rbp-140h]
  unsigned __int16 v84; // [rsp+30h] [rbp-140h]
  __int64 v85; // [rsp+38h] [rbp-138h]
  char v86; // [rsp+38h] [rbp-138h]
  __int64 v88; // [rsp+40h] [rbp-130h]
  unsigned __int16 v89; // [rsp+40h] [rbp-130h]
  __m128i v90; // [rsp+40h] [rbp-130h]
  unsigned __int16 v91; // [rsp+40h] [rbp-130h]
  unsigned __int16 v92; // [rsp+40h] [rbp-130h]
  __int64 v93; // [rsp+50h] [rbp-120h]
  __int128 v95; // [rsp+60h] [rbp-110h]
  int v96; // [rsp+60h] [rbp-110h]
  int v97; // [rsp+68h] [rbp-108h]
  unsigned __int64 v98; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v99; // [rsp+A8h] [rbp-C8h]
  __int64 v100; // [rsp+B0h] [rbp-C0h]
  __int64 v101; // [rsp+B8h] [rbp-B8h]
  __int64 v102; // [rsp+C0h] [rbp-B0h]
  __int64 v103; // [rsp+C8h] [rbp-A8h]
  __int64 v104; // [rsp+D0h] [rbp-A0h]
  __int64 v105; // [rsp+D8h] [rbp-98h]
  unsigned __int128 v106; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v107; // [rsp+F0h] [rbp-80h]
  __int128 v108; // [rsp+100h] [rbp-70h] BYREF
  __int64 v109; // [rsp+110h] [rbp-60h]
  unsigned __int64 v110; // [rsp+120h] [rbp-50h] BYREF
  __int64 v111; // [rsp+128h] [rbp-48h]
  __int64 v112; // [rsp+130h] [rbp-40h]
  __int64 v113; // [rsp+138h] [rbp-38h]

  v6 = 8 * a2;
  v7 = a2;
  v9 = a2 + a1;
  v10 = 8 * v9;
  v79 = a4;
  v11 = *a6;
  v80 = (unsigned int)a4;
  v85 = 16LL * (unsigned int)a4;
  v95 = __PAIR128__(a4, a3);
  v93 = a3;
  v12 = *(_QWORD *)(a3 + 48) + v85;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v110) = v13;
  v111 = v14;
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      goto LABEL_104;
    v16 = 16LL * (v13 - 1);
    v15 = *(_QWORD *)&byte_444C4A0[v16];
    LOBYTE(v16) = byte_444C4A0[v16 + 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v110);
    v102 = v15;
    v103 = v16;
  }
  *(_QWORD *)&v108 = v15;
  BYTE8(v108) = v16;
  LODWORD(v111) = sub_CA1930(&v108);
  v17 = v111;
  if ( (unsigned int)v111 > 0x40 )
  {
    sub_C43690((__int64)&v110, 0, 0);
    if ( v10 == v6 )
      goto LABEL_18;
    if ( v6 <= 0x3F && v10 <= 0x40 )
    {
      v17 = v111;
      v18 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 + 64 - (unsigned __int8)v10) << v6;
      v19 = v110;
      if ( (unsigned int)v111 <= 0x40 )
        goto LABEL_8;
      *(_QWORD *)v110 |= v18;
      goto LABEL_18;
    }
  }
  else
  {
    v110 = 0;
    if ( v10 == v6 )
    {
      v20 = -1;
      goto LABEL_9;
    }
    if ( v10 <= 0x40 && v6 <= 0x3F )
    {
      v18 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 + 64 - (unsigned __int8)v10) << v6;
      v19 = 0;
LABEL_8:
      v20 = ~(v18 | v19);
      goto LABEL_9;
    }
  }
  sub_C43C90(&v110, v6, v10);
LABEL_18:
  v17 = v111;
  if ( (unsigned int)v111 > 0x40 )
  {
    sub_C43D10((__int64)&v110);
    v17 = v111;
    v21 = v110;
    goto LABEL_11;
  }
  v20 = ~v110;
LABEL_9:
  v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & v20;
  if ( !v17 )
    v21 = 0;
LABEL_11:
  v99 = v17;
  v98 = v21;
  v22 = sub_33DD210(v11, v95, *((_QWORD *)&v95 + 1), &v98, 0);
  if ( !v22 )
  {
LABEL_12:
    v23 = 0;
    goto LABEL_13;
  }
  v25 = 8 * a1;
  switch ( v25 )
  {
    case 8:
      v28 = 5;
      goto LABEL_40;
    case 16:
      v28 = 6;
LABEL_40:
      v27 = *(_QWORD *)(v11 + 16);
      if ( !*((_BYTE *)a6 + 34) )
        goto LABEL_41;
      v43 = a6[1];
      goto LABEL_58;
    case 32:
      v28 = 7;
      goto LABEL_40;
    case 64:
      v28 = 8;
      goto LABEL_40;
  }
  v26 = v25 == 128;
  v27 = *(_QWORD *)(v11 + 16);
  if ( v26 )
  {
    if ( !*((_BYTE *)a6 + 34) )
    {
      v28 = 9;
      goto LABEL_41;
    }
    v28 = 9;
    v43 = a6[1];
LABEL_58:
    if ( !*(_QWORD *)(v43 + 8LL * (v28 & 0xF) + 112) )
      goto LABEL_26;
LABEL_41:
    v22 = 0;
    goto LABEL_42;
  }
  v28 = 0;
  if ( !*((_BYTE *)a6 + 34) )
    goto LABEL_41;
LABEL_26:
  v29 = *(unsigned __int16 *)(*(_QWORD *)(v93 + 48) + v85);
  if ( !(_WORD)v29
    || !*(_QWORD *)(v27 + 8 * v29 + 112)
    || !v28
    || *(_BYTE *)((v28 & 0xF) + v27 + 274LL * (unsigned __int16)v29 + 443718) )
  {
    goto LABEL_12;
  }
LABEL_42:
  if ( (*(_WORD *)(a5 + 32) & 0x380) != 0 )
    goto LABEL_12;
  if ( *(_QWORD *)(a5 + 112) )
  {
    v77 = v28;
    v76 = *(_QWORD *)(a5 + 112);
    v88 = v28;
    v30 = sub_2E79000(*(__int64 **)(v11 + 40));
    v31 = sub_2FEBB30(v27, *(_QWORD *)(v11 + 64), v30, v88, 0, v76, 0);
    v28 = v77;
    if ( !v31 )
      goto LABEL_12;
  }
  if ( a2 )
  {
    v62 = *(_QWORD *)(v93 + 80);
    v110 = v62;
    if ( v62 )
    {
      v83 = v28;
      sub_B96E90((__int64)&v110, v62, 1);
      v28 = v83;
    }
    v84 = v28;
    LODWORD(v111) = *(_DWORD *)(v93 + 72);
    *(_QWORD *)&v63 = sub_3400E40(
                        v11,
                        v6,
                        *(unsigned __int16 *)(v85 + *(_QWORD *)(v93 + 48)),
                        *(_QWORD *)(v85 + *(_QWORD *)(v93 + 48) + 8),
                        &v110);
    v64 = sub_3406EB0(
            v11,
            192,
            (unsigned int)&v110,
            *(unsigned __int16 *)(*(_QWORD *)(v93 + 48) + v85),
            *(_QWORD *)(*(_QWORD *)(v93 + 48) + v85 + 8),
            (unsigned int)&v110,
            v95,
            v63);
    v66 = v65;
    v67 = v84;
    v93 = v64;
    *((_QWORD *)&v95 + 1) = v65 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
    v79 = v65;
    if ( v110 )
    {
      sub_B91220((__int64)&v110, v110);
      v67 = v84;
    }
    v92 = v67;
    v68 = (_BYTE *)sub_2E79000(*(__int64 **)(v11 + 40));
    v33 = v92;
    if ( !*v68 )
    {
      v90 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a5 + 40) + 80LL));
      goto LABEL_67;
    }
    v85 = 16LL * v66;
  }
  else
  {
    v89 = v28;
    v32 = (_BYTE *)sub_2E79000(*(__int64 **)(v11 + 40));
    v33 = v89;
    if ( !*v32 )
    {
      v34 = 0;
      v90 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a5 + 40) + 80LL));
      goto LABEL_48;
    }
  }
  v44 = *(_QWORD *)(v93 + 48) + v85;
  v45 = *(_WORD *)v44;
  v46 = *(_QWORD *)(v44 + 8);
  LOWORD(v108) = v45;
  *((_QWORD *)&v108 + 1) = v46;
  if ( !v45 )
  {
    v91 = v33;
    v47 = sub_3007260((__int64)&v108);
    v33 = v91;
    v104 = v47;
    v105 = v48;
    goto LABEL_66;
  }
  if ( v45 == 1 || (unsigned __int16)(v45 - 504) <= 7u )
LABEL_104:
    BUG();
  v48 = 16LL * (v45 - 1) + 71615648;
  v47 = *(_QWORD *)&byte_444C4A0[16 * v45 - 16];
  LOBYTE(v48) = *(_BYTE *)(v48 + 8);
LABEL_66:
  v78 = v33;
  LOBYTE(v111) = v48;
  v110 = (unsigned __int64)(v47 + 7) >> 3;
  v49 = sub_CA1930(&v110);
  v33 = v78;
  v34 = 0;
  v7 = v49 - v9;
  v80 = v79;
  v90 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a5 + 40) + 80LL));
  if ( v49 != v9 )
  {
LABEL_67:
    v50 = *(_QWORD *)(v93 + 80);
    v110 = v50;
    if ( v50 )
    {
      v81 = v33;
      sub_B96E90((__int64)&v110, v50, 1);
      v33 = v81;
    }
    LOBYTE(v101) = 0;
    v34 = v7;
    v82 = v33;
    LODWORD(v111) = *(_DWORD *)(v93 + 72);
    v100 = v7;
    v51 = sub_3409320(v11, v90.m128i_i32[0], v90.m128i_i32[2], v7, v101, (unsigned int)&v110, 0);
    v33 = v82;
    v90.m128i_i64[0] = v51;
    v90.m128i_i64[1] = v52 | v90.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v110 )
    {
      sub_B91220((__int64)&v110, v110);
      v33 = v82;
    }
    v80 = v79;
  }
LABEL_48:
  if ( v22 )
  {
    v110 = 0;
    v111 = 0;
    v35 = v33;
    v36 = *(_QWORD *)(a5 + 112);
    v112 = 0;
    v113 = 0;
    v37 = *(_BYTE *)(v36 + 34);
    v38 = *(_QWORD *)v36 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v38 )
    {
      v71 = *(_QWORD *)(v36 + 8) + v34;
      v72 = *(_BYTE *)(v36 + 20);
      if ( (*(_QWORD *)v36 & 4) != 0 )
      {
        *((_QWORD *)&v108 + 1) = v71;
        BYTE4(v109) = v72;
        *(_QWORD *)&v108 = v38 | 4;
        LODWORD(v109) = *(_DWORD *)(v38 + 12);
      }
      else
      {
        *(_QWORD *)&v108 = *(_QWORD *)v36 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v108 + 1) = v71;
        BYTE4(v109) = v72;
        v73 = *(_QWORD *)(v38 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v73 + 8) - 17 <= 1 )
          v73 = **(_QWORD **)(v73 + 16);
        LODWORD(v109) = *(_DWORD *)(v73 + 8) >> 8;
      }
    }
    else
    {
      v39 = *(_DWORD *)(v36 + 16);
      v40 = *(_QWORD *)(v36 + 8) + v34;
      *(_QWORD *)&v108 = 0;
      *((_QWORD *)&v108 + 1) = v40;
      LODWORD(v109) = v39;
      BYTE4(v109) = 0;
    }
    v41 = *(_QWORD *)(a5 + 80);
    *(_QWORD *)&v106 = v41;
    if ( v41 )
    {
      v86 = v37;
      sub_B96E90((__int64)&v106, v41, 1);
      v37 = v86;
    }
    v42 = *(_QWORD **)(a5 + 40);
    DWORD2(v106) = *(_DWORD *)(a5 + 72);
    v23 = sub_33F5040(
            v11,
            *v42,
            v42[1],
            (unsigned int)&v106,
            v93,
            v80,
            v90.m128i_i64[0],
            v90.m128i_i64[1],
            v108,
            v109,
            v35,
            0,
            v37,
            0,
            (__int64)&v110);
    if ( (_QWORD)v106 )
      sub_B91220((__int64)&v106, v106);
  }
  else
  {
    v53 = v33;
    v54 = *(_QWORD *)(v93 + 80);
    v110 = v54;
    if ( v54 )
      sub_B96E90((__int64)&v110, v54, 1);
    LODWORD(v111) = *(_DWORD *)(v93 + 72);
    *((_QWORD *)&v75 + 1) = v80 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v75 = v93;
    v96 = sub_33FAF80(v11, 216, (unsigned int)&v110, v53, 0, (unsigned int)&v110, v75);
    v97 = v55;
    if ( v110 )
      sub_B91220((__int64)&v110, v110);
    v110 = 0;
    v111 = 0;
    v56 = *(_QWORD *)(a5 + 112);
    v112 = 0;
    v113 = 0;
    v57 = *(_BYTE *)(v56 + 34);
    v58 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v58 )
    {
      v69 = *(_QWORD *)(v56 + 8) + v34;
      v70 = *(_BYTE *)(v56 + 20);
      if ( (*(_QWORD *)v56 & 4) != 0 )
      {
        *((_QWORD *)&v106 + 1) = v69;
        BYTE4(v107) = v70;
        *(_QWORD *)&v106 = v58 | 4;
        LODWORD(v107) = *(_DWORD *)(v58 + 12);
      }
      else
      {
        *(_QWORD *)&v106 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v106 + 1) = v69;
        BYTE4(v107) = v70;
        v74 = *(_QWORD *)(v58 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v74 + 8) - 17 <= 1 )
          v74 = **(_QWORD **)(v74 + 16);
        LODWORD(v107) = *(_DWORD *)(v74 + 8) >> 8;
      }
    }
    else
    {
      v59 = *(_DWORD *)(v56 + 16);
      v60 = *(_QWORD *)(v56 + 8) + v34;
      BYTE4(v107) = 0;
      v106 = __PAIR128__(v60, 0);
      LODWORD(v107) = v59;
    }
    v61 = *(_QWORD *)(a5 + 80);
    *(_QWORD *)&v108 = v61;
    if ( v61 )
      sub_B96E90((__int64)&v108, v61, 1);
    DWORD2(v108) = *(_DWORD *)(a5 + 72);
    v23 = sub_33F4560(
            v11,
            **(_QWORD **)(a5 + 40),
            *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL),
            (unsigned int)&v108,
            v96,
            v97,
            v90.m128i_i64[0],
            v90.m128i_i64[1],
            v106,
            v107,
            v57,
            0,
            (__int64)&v110);
    if ( (_QWORD)v108 )
      sub_B91220((__int64)&v108, v108);
  }
LABEL_13:
  if ( v99 > 0x40 && v98 )
    j_j___libc_free_0_0(v98);
  return v23;
}
