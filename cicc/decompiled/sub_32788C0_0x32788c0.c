// Function: sub_32788C0
// Address: 0x32788c0
//
__int64 __fastcall sub_32788C0(__int64 a1, int a2, __int64 a3, __int64 a4, char a5, int a6)
{
  __int64 *v9; // rax
  int v10; // r15d
  __int64 v11; // rbx
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r8
  unsigned __int16 *v15; // rax
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rax
  int v19; // r9d
  __int64 v20; // r10
  __int64 v21; // r11
  int v22; // edx
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  __int16 v28; // ax
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // r9
  unsigned __int16 *v33; // rdx
  __int64 v34; // r14
  int v35; // eax
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // r14
  __int64 v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // rdx
  unsigned __int64 v51; // r10
  __int64 *v52; // rdx
  __int64 result; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // r14
  __int64 v58; // rax
  int v59; // ecx
  int v60; // r8d
  __int64 v61; // r10
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r11
  __int64 v65; // rbx
  __int64 v66; // rsi
  __int64 v67; // rdx
  __int16 v68; // ax
  __int64 v69; // rdx
  int v70; // eax
  unsigned __int16 *v71; // r8
  __int64 (*v72)(); // rax
  char v73; // al
  __int64 v74; // rax
  __int64 v75; // rdx
  bool v76; // al
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int128 v82; // [rsp-20h] [rbp-1B0h]
  __int128 v83; // [rsp-10h] [rbp-1A0h]
  __int128 v84; // [rsp-10h] [rbp-1A0h]
  __int128 v85; // [rsp-10h] [rbp-1A0h]
  __int128 v86; // [rsp-10h] [rbp-1A0h]
  __int64 v87; // [rsp+0h] [rbp-190h]
  __int64 v88; // [rsp+8h] [rbp-188h]
  unsigned int v89; // [rsp+20h] [rbp-170h]
  __int64 v90; // [rsp+20h] [rbp-170h]
  __int64 v91; // [rsp+28h] [rbp-168h]
  unsigned int v92; // [rsp+38h] [rbp-158h]
  __int128 v93; // [rsp+40h] [rbp-150h]
  __int64 v94; // [rsp+40h] [rbp-150h]
  __int64 v96; // [rsp+50h] [rbp-140h]
  int v97; // [rsp+50h] [rbp-140h]
  __int64 v98; // [rsp+50h] [rbp-140h]
  __int64 v99; // [rsp+50h] [rbp-140h]
  __int64 v100; // [rsp+50h] [rbp-140h]
  __int64 v101; // [rsp+50h] [rbp-140h]
  __int64 v102; // [rsp+58h] [rbp-138h]
  __int64 v103; // [rsp+58h] [rbp-138h]
  __int64 v104; // [rsp+58h] [rbp-138h]
  __int64 v105; // [rsp+60h] [rbp-130h] BYREF
  __int64 v106; // [rsp+68h] [rbp-128h]
  __int64 v107; // [rsp+70h] [rbp-120h] BYREF
  __int64 v108; // [rsp+78h] [rbp-118h]
  __int64 v109; // [rsp+80h] [rbp-110h] BYREF
  int v110; // [rsp+88h] [rbp-108h]
  unsigned __int64 v111; // [rsp+90h] [rbp-100h] BYREF
  unsigned int v112; // [rsp+98h] [rbp-F8h]
  __int64 v113; // [rsp+A0h] [rbp-F0h]
  __int64 v114; // [rsp+A8h] [rbp-E8h]
  __int64 v115; // [rsp+B0h] [rbp-E0h]
  __int64 v116; // [rsp+B8h] [rbp-D8h]
  unsigned __int64 v117; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+C8h] [rbp-C8h]
  _BYTE *v119; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v120; // [rsp+D8h] [rbp-B8h]
  _BYTE v121[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v9 = *(__int64 **)(a1 + 40);
  v10 = *(_DWORD *)(a1 + 24);
  v11 = *v9;
  v12 = *v9;
  v13 = v9[1];
  v14 = *((unsigned int *)v9 + 2);
  v15 = *(unsigned __int16 **)(a1 + 48);
  v16 = *v15;
  v106 = *((_QWORD *)v15 + 1);
  v17 = *(_DWORD *)(v11 + 24);
  LOWORD(v105) = v16;
  if ( v17 == 11 || v17 == 35 )
  {
    *((_QWORD *)&v83 + 1) = v13;
    *(_QWORD *)&v83 = v12;
    return sub_33FAF80(a4, v10, a2, v105, v106, a6, v83);
  }
  if ( v17 == 205 )
  {
    v18 = *(_QWORD *)(v11 + 40);
    v19 = v10;
    v20 = *(_QWORD *)(v18 + 80);
    v21 = *(_QWORD *)(v18 + 88);
    v22 = *(_DWORD *)(*(_QWORD *)(v18 + 40) + 24LL);
    v93 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 40));
    if ( v22 == 35 || v22 == 11 )
    {
      v23 = *(_DWORD *)(*(_QWORD *)(v18 + 80) + 24LL);
      if ( v23 == 35 || v23 == 11 )
      {
        if ( v10 == 214 )
        {
          v71 = (unsigned __int16 *)(*(_QWORD *)(v11 + 48) + 16 * v14);
          v72 = *(__int64 (**)())(*(_QWORD *)a3 + 1432LL);
          if ( v72 != sub_2FE34A0 )
          {
            v90 = v20;
            v91 = v21;
            v73 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, __int64))v72)(
                    a3,
                    *v71,
                    *((_QWORD *)v71 + 1),
                    (unsigned int)v105,
                    v106);
            v19 = 214;
            v20 = v90;
            v21 = v91;
            if ( v73 )
            {
              v16 = (unsigned __int16)v105;
              goto LABEL_8;
            }
          }
        }
        else
        {
          v19 = 213;
          if ( v10 != 215 )
            v19 = v10;
        }
        *((_QWORD *)&v84 + 1) = v21;
        *(_QWORD *)&v84 = v20;
        v97 = v19;
        v54 = sub_33FAF80(a4, v19, a2, v105, v106, v19, v84);
        v56 = v55;
        v57 = v54;
        v58 = sub_33FAF80(a4, v97, a2, v105, v106, v97, v93);
        v59 = v105;
        v60 = v106;
        v61 = v58;
        v62 = *(__int64 **)(v11 + 40);
        v64 = v63;
        v65 = *v62;
        v66 = v62[1];
        v67 = *(_QWORD *)(*v62 + 48) + 16LL * *((unsigned int *)v62 + 2);
        v68 = *(_WORD *)v67;
        v69 = *(_QWORD *)(v67 + 8);
        LOWORD(v119) = v68;
        v120 = v69;
        if ( v68 )
        {
          v70 = ((unsigned __int16)(v68 - 17) < 0xD4u) + 205;
        }
        else
        {
          v100 = v61;
          v103 = v64;
          v76 = sub_30070B0((__int64)&v119);
          v60 = v106;
          v59 = v105;
          v61 = v100;
          v64 = v103;
          v70 = 205 - (!v76 - 1);
        }
        *((_QWORD *)&v85 + 1) = v56;
        *(_QWORD *)&v85 = v57;
        *((_QWORD *)&v82 + 1) = v64;
        *(_QWORD *)&v82 = v61;
        return sub_340EC60(a4, v70, a2, v59, v60, 0, v65, v66, v82, v85);
      }
    }
  }
LABEL_8:
  if ( (_WORD)v16 )
  {
    if ( (unsigned __int16)(v16 - 17) <= 0xD3u )
    {
      v108 = 0;
      v28 = v16;
      LOWORD(v107) = word_4456580[(unsigned __int16)v16 - 1];
      v16 = (unsigned __int16)v107;
      goto LABEL_56;
    }
    goto LABEL_10;
  }
  v16 = (unsigned int)v16;
  if ( !sub_30070B0((__int64)&v105) )
  {
LABEL_10:
    v27 = v106;
    v28 = v16;
    goto LABEL_11;
  }
  v16 = (unsigned int)sub_3009970((__int64)&v105, (unsigned int)v16, v24, v25, v26);
  v28 = v105;
LABEL_11:
  LOWORD(v107) = v16;
  v108 = v27;
  if ( !v28 )
  {
    v16 = (unsigned int)v16;
    if ( sub_30070B0((__int64)&v105) )
      goto LABEL_13;
    return 0;
  }
LABEL_56:
  if ( (unsigned __int16)(v28 - 17) > 0xD3u )
    return 0;
LABEL_13:
  if ( a5 )
  {
    if ( !(_WORD)v16 )
      return 0;
    v16 = (unsigned __int16)v16;
    if ( !*(_QWORD *)(a3 + 8LL * (unsigned __int16)v16 + 112) )
      return 0;
  }
  if ( !(unsigned __int8)sub_33CA6D0(v11) )
    return 0;
  if ( (_WORD)v107 )
  {
    if ( (_WORD)v107 == 1 || (unsigned __int16)(v107 - 504) <= 7u )
      goto LABEL_90;
    v30 = 16LL * ((unsigned __int16)v107 - 1);
    v29 = *(_QWORD *)&byte_444C4A0[v30];
    LOBYTE(v30) = byte_444C4A0[v30 + 8];
  }
  else
  {
    v29 = sub_3007260((__int64)&v107);
    v113 = v29;
    v114 = v30;
  }
  v119 = (_BYTE *)v29;
  LOBYTE(v120) = v30;
  v31 = sub_CA1930(&v119);
  v33 = *(unsigned __int16 **)(v11 + 48);
  v89 = v31;
  v34 = *((_QWORD *)v33 + 1);
  v35 = *v33;
  v118 = v34;
  LOWORD(v117) = v35;
  if ( (_WORD)v35 )
  {
    if ( (unsigned __int16)(v35 - 17) > 0xD3u )
    {
      LOWORD(v119) = v35;
      v120 = v34;
LABEL_22:
      if ( (_WORD)v35 != 1 && (unsigned __int16)(v35 - 504) > 7u )
      {
        v36 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v35 - 16];
        goto LABEL_25;
      }
LABEL_90:
      BUG();
    }
    LOWORD(v35) = word_4456580[v35 - 1];
    v81 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v117) )
    {
      v120 = v34;
      LOWORD(v119) = 0;
      goto LABEL_78;
    }
    LOWORD(v35) = sub_3009970((__int64)&v117, v16, v77, v78, v79);
  }
  LOWORD(v119) = v35;
  v120 = v81;
  if ( (_WORD)v35 )
    goto LABEL_22;
LABEL_78:
  v36 = sub_3007260((__int64)&v119);
  v115 = v36;
  v116 = v80;
LABEL_25:
  v92 = v36;
  v119 = v121;
  v120 = 0x800000000LL;
  if ( !(_WORD)v105 )
  {
    if ( !sub_3007100((__int64)&v105) )
      goto LABEL_75;
    goto LABEL_79;
  }
  if ( (unsigned __int16)(v105 - 176) <= 0x34u )
  {
LABEL_79:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v105 )
    {
      if ( (unsigned __int16)(v105 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_27;
    }
LABEL_75:
    v37 = sub_3007130((__int64)&v105, v16);
    goto LABEL_28;
  }
LABEL_27:
  v37 = word_4456340[(unsigned __int16)v105 - 1];
LABEL_28:
  if ( v37 )
  {
    v38 = 0;
    v94 = 40LL * v37;
    while ( 1 )
    {
      v40 = *(_QWORD *)(*(_QWORD *)(v11 + 40) + v38);
      if ( *(_DWORD *)(v40 + 24) != 51 )
      {
        v39 = *(_QWORD *)(v40 + 80);
        v109 = v39;
        if ( v39 )
        {
          v96 = v40;
          sub_B96E90((__int64)&v109, v39, 1);
          v40 = v96;
        }
        v41 = *(_QWORD *)(v40 + 96);
        v110 = *(_DWORD *)(v40 + 72);
        sub_C44AB0((__int64)&v111, v41 + 24, v92);
        if ( v10 == 213 || v10 == 224 )
          sub_C44830((__int64)&v117, &v111, v89);
        else
          sub_C449B0((__int64)&v117, (const void **)&v111, v89);
        v42 = sub_34007B0(a4, (unsigned int)&v117, (unsigned int)&v109, v107, v108, 0, 0);
        v32 = v43;
        v44 = (unsigned int)v120;
        v45 = v42;
        if ( (unsigned __int64)(unsigned int)v120 + 1 > HIDWORD(v120) )
        {
          v101 = v42;
          v104 = v32;
          sub_C8D5F0((__int64)&v119, v121, (unsigned int)v120 + 1LL, 0x10u, v42, v32);
          v44 = (unsigned int)v120;
          v45 = v101;
          v32 = v104;
        }
        v46 = (__int64 *)&v119[16 * v44];
        *v46 = v45;
        v46[1] = v32;
        LODWORD(v120) = v120 + 1;
        if ( (unsigned int)v118 > 0x40 && v117 )
          j_j___libc_free_0_0(v117);
        if ( v112 > 0x40 && v111 )
          j_j___libc_free_0_0(v111);
        if ( v109 )
          sub_B91220((__int64)&v109, v109);
        goto LABEL_45;
      }
      if ( (v10 & 0xFFFFFFF7) == 0xD7 )
        break;
      v74 = sub_3400BD0(a4, 0, a2, v107, v108, 0, 0);
      v32 = v75;
      v50 = (unsigned int)v120;
      v49 = v74;
      v51 = (unsigned int)v120 + 1LL;
      if ( v51 > HIDWORD(v120) )
        goto LABEL_69;
LABEL_51:
      v52 = (__int64 *)&v119[16 * v50];
      *v52 = v49;
      v52[1] = v32;
      LODWORD(v120) = v120 + 1;
LABEL_45:
      v38 += 40;
      if ( v94 == v38 )
        goto LABEL_71;
    }
    v117 = 0;
    LODWORD(v118) = 0;
    v47 = sub_33F17F0(a4, 51, &v117, v107, v108);
    v49 = v47;
    v32 = v48;
    if ( v117 )
    {
      v87 = v47;
      v88 = v48;
      sub_B91220((__int64)&v117, v117);
      v49 = v87;
      v32 = v88;
    }
    v50 = (unsigned int)v120;
    v51 = (unsigned int)v120 + 1LL;
    if ( v51 <= HIDWORD(v120) )
      goto LABEL_51;
LABEL_69:
    v98 = v49;
    v102 = v32;
    sub_C8D5F0((__int64)&v119, v121, v51, 0x10u, v49, v32);
    v50 = (unsigned int)v120;
    v49 = v98;
    v32 = v102;
    goto LABEL_51;
  }
LABEL_71:
  *((_QWORD *)&v86 + 1) = (unsigned int)v120;
  *(_QWORD *)&v86 = v119;
  result = sub_33FC220(a4, 156, a2, v105, v106, v32, v86);
  if ( v119 != v121 )
  {
    v99 = result;
    _libc_free((unsigned __int64)v119);
    return v99;
  }
  return result;
}
