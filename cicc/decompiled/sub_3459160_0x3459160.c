// Function: sub_3459160
// Address: 0x3459160
//
__int64 __fastcall sub_3459160(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // r13
  const __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned int v13; // r9d
  unsigned __int16 v14; // r13
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 (__fastcall *v17)(_BYTE *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rdx
  int v21; // r9d
  __int128 v22; // rax
  unsigned __int8 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // r15
  __int128 v26; // rax
  __int64 v27; // r9
  __int64 v28; // r14
  unsigned int v29; // edx
  unsigned int v30; // ebx
  unsigned __int8 *v31; // rax
  unsigned int v32; // esi
  __int64 v33; // r15
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // r10
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // r11
  __int16 v41; // ax
  unsigned int v42; // r14d
  __int64 v43; // r12
  unsigned int v44; // r11d
  __int64 v45; // r10
  __int128 v46; // rax
  __int64 v47; // r9
  unsigned __int8 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r15
  unsigned __int8 *v51; // r14
  __int128 v52; // rax
  __int64 v53; // r9
  __int64 v54; // rcx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // rsi
  bool v58; // al
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r15
  bool v64; // al
  __int64 v65; // rax
  char v66; // al
  char v67; // al
  __int64 v68; // r10
  unsigned int v69; // r11d
  char v70; // al
  unsigned __int8 *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r15
  unsigned __int8 *v74; // r14
  __int128 v75; // rax
  __int64 v76; // r9
  char v77; // al
  char v78; // al
  __int128 v79; // [rsp-20h] [rbp-E0h]
  __int128 v80; // [rsp-10h] [rbp-D0h]
  __int128 v81; // [rsp+0h] [rbp-C0h]
  __int128 v82; // [rsp+10h] [rbp-B0h]
  unsigned int v83; // [rsp+10h] [rbp-B0h]
  __int64 v84; // [rsp+10h] [rbp-B0h]
  unsigned int v85; // [rsp+10h] [rbp-B0h]
  unsigned int v86; // [rsp+10h] [rbp-B0h]
  unsigned int v87; // [rsp+24h] [rbp-9Ch]
  unsigned int v88; // [rsp+24h] [rbp-9Ch]
  unsigned int v89; // [rsp+24h] [rbp-9Ch]
  unsigned int v90; // [rsp+24h] [rbp-9Ch]
  __int64 v91; // [rsp+28h] [rbp-98h]
  unsigned int v92; // [rsp+28h] [rbp-98h]
  __int64 v93; // [rsp+28h] [rbp-98h]
  unsigned int v94; // [rsp+28h] [rbp-98h]
  __int64 v95; // [rsp+28h] [rbp-98h]
  __int64 v96; // [rsp+28h] [rbp-98h]
  unsigned int v97; // [rsp+28h] [rbp-98h]
  __int64 v98; // [rsp+30h] [rbp-90h]
  __int64 v99; // [rsp+38h] [rbp-88h]
  unsigned __int128 v100; // [rsp+40h] [rbp-80h]
  __int64 v101; // [rsp+50h] [rbp-70h] BYREF
  int v102; // [rsp+58h] [rbp-68h]
  __int64 v103; // [rsp+60h] [rbp-60h] BYREF
  __int64 v104; // [rsp+68h] [rbp-58h]
  __int64 v105; // [rsp+70h] [rbp-50h]
  __int64 v106; // [rsp+78h] [rbp-48h]
  __int16 v107; // [rsp+80h] [rbp-40h] BYREF
  __int64 v108; // [rsp+88h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v101 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v101, v5, 1);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v102 = *(_DWORD *)(a2 + 72);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = *(const __m128i **)(a2 + 40);
  LOWORD(v103) = v7;
  v104 = v8;
  v10 = _mm_loadu_si128(v9);
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      v107 = v7;
      v108 = v8;
      goto LABEL_6;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v11 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v103) )
    {
      v108 = v8;
      v107 = 0;
      goto LABEL_11;
    }
    LOWORD(v7) = sub_3009970((__int64)&v103, v5, v60, v61, v62);
  }
  v107 = v7;
  v108 = v11;
  if ( (_WORD)v7 )
  {
LABEL_6:
    if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v98 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
    goto LABEL_12;
  }
LABEL_11:
  v105 = sub_3007260((__int64)&v107);
  v106 = v12;
  LODWORD(v98) = v105;
LABEL_12:
  v13 = v98;
  v14 = v103;
  if ( *(_DWORD *)(a2 + 24) == 203 )
  {
    v63 = 1;
    if ( (_WORD)v103 == 1 )
      goto LABEL_38;
    if ( !(_WORD)v103 )
      goto LABEL_29;
    v63 = (unsigned __int16)v103;
    if ( *(_QWORD *)&a1[8 * (unsigned __int16)v103 + 112] )
    {
LABEL_38:
      if ( (a1[500 * v63 + 6612] & 0xFB) == 0 )
      {
        v55 = (unsigned int)v103;
        v57 = 198;
        v56 = v104;
        goto LABEL_40;
      }
    }
  }
  v15 = 1;
  if ( (_WORD)v103 != 1 )
  {
    if ( (_WORD)v103 )
    {
      v15 = (unsigned __int16)v103;
      if ( !*(_QWORD *)&a1[8 * (unsigned __int16)v103 + 112] )
        goto LABEL_20;
      goto LABEL_16;
    }
LABEL_29:
    v58 = sub_30070B0((__int64)&v103);
    v13 = v98;
    if ( !v58 )
    {
      v44 = v103;
      v45 = v104;
      goto LABEL_46;
    }
LABEL_30:
    if ( !v13 || (v13 & (v13 - 1)) != 0 )
      goto LABEL_31;
    v15 = 1;
    if ( v14 != 1 )
    {
      if ( !v14 )
      {
        v90 = v13;
        v86 = v103;
        v96 = v104;
        v78 = sub_328A020((__int64)a1, 0xC7u, v103, v104, 0);
        v68 = v96;
        v69 = v86;
        v13 = v90;
        if ( v78 )
          goto LABEL_31;
        goto LABEL_74;
      }
      v15 = v14;
      if ( !*(_QWORD *)&a1[8 * v14 + 112] )
      {
        v89 = v13;
        v85 = v103;
        v95 = v104;
        v77 = sub_328A020((__int64)a1, 0xC7u, v103, v104, 0);
        v68 = v95;
        v69 = v85;
        v13 = v89;
        if ( v77 )
          goto LABEL_70;
        goto LABEL_74;
      }
    }
    v66 = a1[500 * v15 + 6614];
    if ( !v66
      || v66 == 4
      || (v87 = v13,
          v83 = v103,
          v93 = v104,
          v67 = sub_328A020((__int64)a1, 0xC7u, v103, v104, 0),
          v68 = v93,
          v69 = v83,
          v13 = v87,
          v67) )
    {
LABEL_53:
      v15 = 1;
      if ( v14 == 1 )
      {
LABEL_54:
        v94 = v13;
        if ( (a1[500 * (unsigned int)v15 + 6471] & 0xFB) == 0 )
        {
          v88 = v103;
          v84 = v104;
          if ( (unsigned __int8)sub_328C7F0((__int64)a1, 0xBAu, v103, v104, 0) )
          {
            v70 = sub_328C7F0((__int64)a1, 0xBCu, v88, v84, 0);
            v45 = v84;
            v44 = v88;
            v13 = v94;
            if ( v70 )
            {
              if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
                goto LABEL_24;
              goto LABEL_22;
            }
          }
        }
LABEL_31:
        v43 = 0;
        goto LABEL_32;
      }
      if ( !v14 )
        goto LABEL_31;
      v15 = v14;
LABEL_70:
      if ( !*(_QWORD *)&a1[8 * (int)v15 + 112] )
        goto LABEL_31;
      goto LABEL_54;
    }
LABEL_74:
    v97 = v13;
    if ( !sub_34447D0((__int64)a1, v69, v68) )
      goto LABEL_31;
    v14 = v103;
    v13 = v97;
    goto LABEL_53;
  }
LABEL_16:
  if ( (a1[500 * (unsigned int)v15 + 6617] & 0xFB) == 0 )
  {
    v16 = *(_QWORD *)(a3 + 64);
    v17 = *(__int64 (__fastcall **)(_BYTE *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
    v18 = sub_2E79000(*(__int64 **)(a3 + 40));
    v19 = v17(a1, v18, v16, (unsigned int)v103, v104);
    v91 = v20;
    *(_QWORD *)&v22 = sub_33FAF80(a3, 203, (__int64)&v101, (unsigned int)v103, v104, v21, v10);
    v82 = v22;
    v23 = sub_3400BD0(a3, 0, (__int64)&v101, (unsigned int)v103, v104, 0, v10, 0);
    v25 = v24;
    *(_QWORD *)&v26 = sub_33ED040((_QWORD *)a3, 0x11u);
    *((_QWORD *)&v79 + 1) = v25;
    *(_QWORD *)&v79 = v23;
    v28 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v101, v19, v91, v27, *(_OWORD *)&v10, v79, v26);
    v30 = v29;
    v31 = sub_3400BD0(a3, (unsigned int)v98, (__int64)&v101, (unsigned int)v103, v104, 0, v10, 0);
    v32 = v103;
    v33 = v104;
    v35 = v34;
    v36 = (unsigned __int64)v31;
    v37 = v28;
    v38 = v30;
    v39 = *(_QWORD *)(v28 + 48) + 16LL * v30;
    v40 = v38;
    v41 = *(_WORD *)v39;
    v108 = *(_QWORD *)(v39 + 8);
    v107 = v41;
    if ( v41 )
    {
      v42 = ((unsigned __int16)(v41 - 17) < 0xD4u) + 205;
    }
    else
    {
      v92 = v103;
      v99 = v40;
      v100 = __PAIR128__(v35, v36);
      v64 = sub_30070B0((__int64)&v107);
      v32 = v92;
      v37 = v28;
      v40 = v99;
      v35 = *((_QWORD *)&v100 + 1);
      v36 = v100;
      v42 = 205 - (!v64 - 1);
    }
    v43 = sub_340EC60((_QWORD *)a3, v42, (__int64)&v101, v32, v33, 0, v37, v40, __PAIR128__(v35, v36), v82);
    goto LABEL_32;
  }
LABEL_20:
  if ( (unsigned __int16)(v103 - 17) <= 0xD3u )
    goto LABEL_30;
  v44 = v103;
  v45 = v104;
LABEL_22:
  if ( *(_QWORD *)&a1[8 * (int)v15 + 112] )
  {
    if ( a1[500 * (unsigned int)v15 + 6614] != 2 )
      goto LABEL_24;
LABEL_60:
    if ( !a1[500 * v15 + 6613] )
      goto LABEL_24;
    goto LABEL_46;
  }
  if ( v14 == 1 )
    goto LABEL_60;
LABEL_46:
  v65 = sub_34587B0((__int64)a1, a2, a3, (__int64)&v101, v44, v45, v10, *(_OWORD *)&v10, v13);
  if ( v65 )
  {
    v43 = v65;
    goto LABEL_32;
  }
  v44 = v103;
  v45 = v104;
LABEL_24:
  *(_QWORD *)&v46 = sub_3400BD0(a3, 1, (__int64)&v101, v44, v45, 0, v10, 0);
  v48 = sub_3406EB0((_QWORD *)a3, 0x39u, (__int64)&v101, (unsigned int)v103, v104, v47, *(_OWORD *)&v10, v46);
  v50 = v49;
  v51 = v48;
  *(_QWORD *)&v52 = sub_34074A0((_QWORD *)a3, (__int64)&v101, v10.m128i_i64[0], v10.m128i_i64[1], v103, v104, v10);
  *((_QWORD *)&v81 + 1) = v50;
  *(_QWORD *)&v81 = v51;
  sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v101, (unsigned int)v103, v104, v53, v52, v81);
  if ( (_WORD)v103 == 1 )
  {
    if ( a1[7113] )
    {
LABEL_28:
      v55 = (unsigned int)v103;
      v56 = v104;
      v57 = 200;
LABEL_40:
      v43 = (__int64)sub_33FAF80(a3, v57, (__int64)&v101, v55, v56, v13, v10);
      goto LABEL_32;
    }
    v54 = 1;
  }
  else
  {
    if ( !(_WORD)v103 )
      goto LABEL_28;
    if ( !*(_QWORD *)&a1[8 * (unsigned __int16)v103 + 112] )
      goto LABEL_28;
    v54 = (unsigned __int16)v103;
    if ( a1[500 * (unsigned __int16)v103 + 6613] )
      goto LABEL_28;
  }
  if ( !a1[500 * v54 + 6614] )
    goto LABEL_28;
  v71 = sub_33FAF80(a3, 199, (__int64)&v101, (unsigned int)v103, v104, v13, v10);
  v73 = v72;
  v74 = v71;
  *(_QWORD *)&v75 = sub_3400BD0(a3, (unsigned int)v98, (__int64)&v101, (unsigned int)v103, v104, 0, v10, 0);
  *((_QWORD *)&v80 + 1) = v73;
  *(_QWORD *)&v80 = v74;
  v43 = (__int64)sub_3406EB0((_QWORD *)a3, 0x39u, (__int64)&v101, (unsigned int)v103, v104, v76, v75, v80);
LABEL_32:
  if ( v101 )
    sub_B91220((__int64)&v101, v101);
  return v43;
}
