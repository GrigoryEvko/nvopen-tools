// Function: sub_1297D00
// Address: 0x1297d00
//
__int64 __fastcall sub_1297D00(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6, char a7)
{
  __int64 v8; // r14
  unsigned int v9; // ebx
  const char *v10; // r12
  __int64 v11; // r13
  unsigned int v12; // eax
  bool v13; // cc
  unsigned __int64 v14; // rsi
  _QWORD *v15; // r13
  _QWORD *v16; // rax
  __int64 v17; // r8
  __int64 result; // rax
  bool v19; // al
  __int64 v20; // r9
  _QWORD *v21; // r13
  __int64 v22; // rax
  size_t v23; // rax
  __int64 v24; // rcx
  unsigned int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r9
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rdx
  _BYTE *v38; // rsi
  _QWORD *v39; // r10
  __int64 v40; // r8
  __int64 v41; // r13
  const void *v42; // r14
  _DWORD *v43; // r12
  size_t v44; // rbx
  size_t v45; // r15
  size_t v46; // rdx
  int v47; // eax
  _DWORD *v48; // rcx
  size_t v49; // r9
  const void **v50; // rax
  size_t v51; // rcx
  size_t v52; // rdx
  int v53; // eax
  __int64 v54; // r9
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // r10
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // r10
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r9
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rsi
  size_t v75; // rdx
  __int64 v76; // rcx
  unsigned int v77; // eax
  __int64 v78; // [rsp+0h] [rbp-E0h]
  unsigned int v79; // [rsp+8h] [rbp-D8h]
  __int64 v80; // [rsp+8h] [rbp-D8h]
  __int64 *v81; // [rsp+10h] [rbp-D0h]
  __int64 v82; // [rsp+10h] [rbp-D0h]
  __int64 *v83; // [rsp+10h] [rbp-D0h]
  __int64 v84; // [rsp+10h] [rbp-D0h]
  __int64 v85; // [rsp+10h] [rbp-D0h]
  __int64 v86; // [rsp+10h] [rbp-D0h]
  __int64 v87; // [rsp+10h] [rbp-D0h]
  __int64 *v88; // [rsp+10h] [rbp-D0h]
  __int64 v89; // [rsp+10h] [rbp-D0h]
  unsigned int v90; // [rsp+18h] [rbp-C8h]
  __int64 v91; // [rsp+18h] [rbp-C8h]
  __int64 v92; // [rsp+18h] [rbp-C8h]
  __int64 v93; // [rsp+18h] [rbp-C8h]
  __int64 v94; // [rsp+18h] [rbp-C8h]
  __int64 v95; // [rsp+18h] [rbp-C8h]
  size_t v96; // [rsp+18h] [rbp-C8h]
  unsigned int v97; // [rsp+18h] [rbp-C8h]
  __int64 v98; // [rsp+18h] [rbp-C8h]
  const char *v99; // [rsp+28h] [rbp-B8h]
  size_t v100; // [rsp+28h] [rbp-B8h]
  __int64 v101; // [rsp+28h] [rbp-B8h]
  __int64 v102; // [rsp+28h] [rbp-B8h]
  unsigned int v104; // [rsp+38h] [rbp-A8h]
  __int64 v105; // [rsp+38h] [rbp-A8h]
  __int64 v106; // [rsp+38h] [rbp-A8h]
  __int64 v107; // [rsp+38h] [rbp-A8h]
  __int64 v108; // [rsp+38h] [rbp-A8h]
  __int64 v109; // [rsp+38h] [rbp-A8h]
  _QWORD *v110; // [rsp+40h] [rbp-A0h]
  _QWORD *v111; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v114; // [rsp+58h] [rbp-88h]
  __int64 v115; // [rsp+68h] [rbp-78h] BYREF
  __int64 v116[2]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v117[2]; // [rsp+80h] [rbp-60h] BYREF
  void *s2; // [rsp+90h] [rbp-50h] BYREF
  size_t v119; // [rsp+98h] [rbp-48h]
  _QWORD v120[8]; // [rsp+A0h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
    sub_15E08E0(a3);
  v114 = *(_QWORD *)(a3 + 88);
  if ( sub_1297BB0(a1[4], a2) )
  {
    s2 = "agg.result";
    LOWORD(v120[0]) = 259;
    sub_164B780(v114, &s2);
    v114 += 40LL;
  }
  v8 = *(_QWORD *)(a2 + 16) + 40LL;
  if ( a4 )
  {
    v9 = 1;
    while ( 1 )
    {
      v10 = *(const char **)(a4 + 8);
      v11 = *(_QWORD *)(v8 + 24);
      if ( v10 )
        break;
      v10 = "temp_param";
      if ( (*(_BYTE *)(a4 + 172) & 1) != 0 )
        v10 = "this";
      v12 = *(_DWORD *)(v8 + 12);
      v13 = v12 <= 2;
      if ( v12 == 2 )
      {
LABEL_28:
        if ( !a7 || !*(_BYTE *)(v8 + 16) )
          goto LABEL_32;
        if ( !unk_4D046B0 )
          goto LABEL_31;
        v38 = (_BYTE *)sub_1649960(a3);
        if ( v38 )
        {
          s2 = v120;
          sub_1297340((__int64 *)&s2, v38, (__int64)&v38[v37]);
          v39 = s2;
          v40 = *(_QWORD *)&dword_4D04688[2];
          if ( !*(_QWORD *)&dword_4D04688[2] )
            goto LABEL_112;
        }
        else
        {
          v39 = v120;
          v119 = 0;
          v40 = *(_QWORD *)&dword_4D04688[2];
          s2 = v120;
          LOBYTE(v120[0]) = 0;
          if ( !*(_QWORD *)&dword_4D04688[2] )
            goto LABEL_32;
        }
        v99 = v10;
        v95 = v11;
        v41 = v40;
        v82 = v8;
        v42 = v39;
        v79 = v9;
        v43 = dword_4D04688;
        v44 = v119;
        v78 = a4;
        do
        {
          while ( 1 )
          {
            v45 = *(_QWORD *)(v41 + 40);
            v46 = v44;
            if ( v45 <= v44 )
              v46 = *(_QWORD *)(v41 + 40);
            if ( v46 )
            {
              v47 = memcmp(*(const void **)(v41 + 32), v42, v46);
              if ( v47 )
                break;
            }
            if ( (__int64)(v45 - v44) >= 0x80000000LL )
              goto LABEL_67;
            if ( (__int64)(v45 - v44) > (__int64)0xFFFFFFFF7FFFFFFFLL )
            {
              v47 = v45 - v44;
              break;
            }
LABEL_58:
            v41 = *(_QWORD *)(v41 + 24);
            if ( !v41 )
              goto LABEL_68;
          }
          if ( v47 < 0 )
            goto LABEL_58;
LABEL_67:
          v43 = (_DWORD *)v41;
          v41 = *(_QWORD *)(v41 + 16);
        }
        while ( v41 );
LABEL_68:
        v48 = v43;
        v49 = v44;
        v39 = v42;
        v11 = v95;
        v9 = v79;
        v10 = v99;
        v8 = v82;
        a4 = v78;
        if ( v48 == dword_4D04688 )
          goto LABEL_112;
        v50 = (const void **)v48;
        v51 = *((_QWORD *)v48 + 5);
        v52 = v49;
        if ( v51 <= v49 )
          v52 = v51;
        if ( v52 )
        {
          v96 = v49;
          v100 = v51;
          v111 = v39;
          v53 = memcmp(v39, v50[4], v52);
          v39 = v111;
          v51 = v100;
          v49 = v96;
          if ( v53 )
          {
LABEL_76:
            if ( v53 < 0 )
              goto LABEL_112;
LABEL_77:
            if ( v39 != v120 )
              j_j___libc_free_0(v39, v120[0] + 1LL);
LABEL_31:
            if ( *(_BYTE *)(v8 + 33) )
              goto LABEL_32;
            v104 = unk_4D0463C;
            if ( unk_4D0463C )
              v104 = sub_126A420(a1[4], v114);
            if ( *(char *)(v11 + 142) >= 0 && *(_BYTE *)(v11 + 140) == 12 )
              v97 = sub_8D4AB0(v11);
            else
              v97 = *(_DWORD *)(v11 + 136);
            s2 = "tmp";
            LOWORD(v120[0]) = 259;
            v55 = sub_127FE40(a1, v11, (__int64)&s2);
            LOWORD(v120[0]) = 257;
            v21 = v55;
            v56 = sub_1648A60(64, 1);
            v57 = v56;
            if ( v56 )
            {
              v101 = v56;
              sub_15F9210(v56, *(_QWORD *)(*(_QWORD *)v114 + 24LL), v114, 0, v104, 0);
              v57 = v101;
            }
            v58 = a1[7];
            if ( v58 )
            {
              v102 = v57;
              v83 = (__int64 *)a1[8];
              sub_157E9D0(v58 + 40, v57);
              v57 = v102;
              v59 = *(_QWORD *)(v102 + 24);
              v60 = *v83;
              *(_QWORD *)(v102 + 32) = v83;
              v60 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v102 + 24) = v60 | v59 & 7;
              *(_QWORD *)(v60 + 8) = v102 + 24;
              *v83 = *v83 & 7 | (v102 + 24);
            }
            v84 = v57;
            sub_164B780(v57, &s2);
            v61 = v84;
            v62 = a1[6];
            if ( v62 )
            {
              v116[0] = a1[6];
              sub_1623A60(v116, v62, 2);
              v61 = v84;
              v63 = v84 + 48;
              if ( *(_QWORD *)(v84 + 48) )
              {
                v80 = v84;
                v85 = v84 + 48;
                sub_161E7C0(v85);
                v61 = v80;
                v63 = v85;
              }
              v64 = v116[0];
              *(_QWORD *)(v61 + 48) = v116[0];
              if ( v64 )
              {
                v86 = v61;
                sub_1623210(v116, v64, v63);
                v61 = v86;
              }
            }
            v87 = v61;
            sub_15F8F50(v61, v97);
            LOWORD(v120[0]) = 257;
            v65 = sub_1648A60(64, 2);
            v66 = v65;
            if ( v65 )
            {
              v67 = v104;
              v105 = v65;
              sub_15F9650(v65, v87, v21, v67, 0);
              v66 = v105;
            }
            v68 = a1[7];
            if ( v68 )
            {
              v106 = v66;
              v88 = (__int64 *)a1[8];
              sub_157E9D0(v68 + 40, v66);
              v66 = v106;
              v69 = *(_QWORD *)(v106 + 24);
              v70 = *v88;
              *(_QWORD *)(v106 + 32) = v88;
              v70 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v106 + 24) = v70 | v69 & 7;
              *(_QWORD *)(v70 + 8) = v106 + 24;
              *v88 = *v88 & 7 | (v106 + 24);
            }
            v107 = v66;
            sub_164B780(v66, &s2);
            v71 = v107;
            v72 = a1[6];
            if ( v72 )
            {
              v116[0] = a1[6];
              sub_1623A60(v116, v72, 2);
              v71 = v107;
              v73 = v107 + 48;
              if ( *(_QWORD *)(v107 + 48) )
              {
                v89 = v107;
                v108 = v107 + 48;
                sub_161E7C0(v108);
                v71 = v89;
                v73 = v108;
              }
              v74 = v116[0];
              *(_QWORD *)(v71 + 48) = v116[0];
              if ( v74 )
              {
                v109 = v71;
                sub_1623210(v116, v74, v73);
                v71 = v109;
              }
            }
            sub_15F9450(v71, v97);
            v116[0] = (__int64)v117;
            sub_1297340(v116, "__val_param", (__int64)"");
            v75 = strlen(v10);
            if ( v75 <= 0x3FFFFFFFFFFFFFFFLL - v116[1] )
            {
              sub_2241490(v116, v10, v75, v76);
              LOWORD(v120[0]) = 260;
              s2 = v116;
              sub_164B780(v114, &s2);
              if ( (_QWORD *)v116[0] != v117 )
                j_j___libc_free_0(v116[0], v117[0] + 1LL);
              v20 = *(unsigned __int8 *)(v8 + 33);
              goto LABEL_34;
            }
LABEL_115:
            sub_4262D8((__int64)"basic_string::append");
          }
        }
        v54 = v49 - v51;
        if ( v54 >= 0x80000000LL )
          goto LABEL_77;
        if ( v54 >= (__int64)0xFFFFFFFF80000000LL )
        {
          v53 = v54;
          goto LABEL_76;
        }
LABEL_112:
        if ( v39 != v120 )
          j_j___libc_free_0(v39, v120[0] + 1LL);
LABEL_32:
        v19 = sub_127B420(v11);
        v20 = *(unsigned __int8 *)(v8 + 33);
        v21 = (_QWORD *)v114;
        if ( !v19 && !(_BYTE)v20 )
          sub_127B550("Non-aggregate arguments passed indirectly are not supported!", a6, 1);
LABEL_34:
        sub_12A38F0(a1, a4, v10, v21, v9, v20);
LABEL_13:
        a4 = *(_QWORD *)(a4 + 112);
        v114 += 40LL;
        v8 += 40;
        ++v9;
        if ( !a4 )
          goto LABEL_21;
      }
      else
      {
LABEL_16:
        if ( v13 )
        {
          if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
            sub_15E08E0(a3);
          if ( v114 == *(_QWORD *)(a3 + 88) + 40LL * *(_QWORD *)(a3 + 96) )
            sub_127B550("Argument mismatch in generation function prolog!", a6, 1);
          v119 = 0;
          s2 = v120;
          LOBYTE(v120[0]) = 0;
          v110 = (_QWORD *)v114;
          if ( sub_127B420(v11) )
          {
            v23 = strlen(v10);
            sub_2241130(&s2, 0, v119, v10, v23);
            if ( 0x3FFFFFFFFFFFFFFFLL - v119 <= 4 )
              goto LABEL_115;
            sub_2241490(&s2, ".addr", 5, v24);
            v116[0] = (__int64)"tmp";
            LOWORD(v117[0]) = 259;
            v110 = sub_127FE40(a1, v11, (__int64)v116);
            v25 = unk_4D0463C;
            if ( unk_4D0463C )
              v25 = sub_126A420(a1[4], (unsigned __int64)v110);
            v90 = v25;
            LOWORD(v117[0]) = 257;
            v26 = sub_1648A60(64, 2);
            v27 = v26;
            if ( v26 )
            {
              v28 = v90;
              v91 = v26;
              sub_15F9650(v26, v114, v110, v28, 0);
              v27 = v91;
            }
            v29 = a1[7];
            if ( v29 )
            {
              v92 = v27;
              v81 = (__int64 *)a1[8];
              sub_157E9D0(v29 + 40, v27);
              v27 = v92;
              v30 = *(_QWORD *)(v92 + 24);
              v31 = *v81;
              *(_QWORD *)(v92 + 32) = v81;
              v31 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v92 + 24) = v31 | v30 & 7;
              *(_QWORD *)(v31 + 8) = v92 + 24;
              *v81 = *v81 & 7 | (v92 + 24);
            }
            v93 = v27;
            sub_164B780(v27, v116);
            v32 = v93;
            v33 = a1[6];
            if ( v33 )
            {
              v115 = a1[6];
              sub_1623A60(&v115, v33, 2);
              v32 = v93;
              v34 = v93 + 48;
              if ( *(_QWORD *)(v93 + 48) )
              {
                sub_161E7C0(v93 + 48);
                v32 = v93;
                v34 = v93 + 48;
              }
              v35 = v115;
              *(_QWORD *)(v32 + 48) = v115;
              if ( v35 )
              {
                v94 = v32;
                sub_1623210(&v115, v35, v34);
                v32 = v94;
              }
            }
            if ( *(char *)(v11 + 142) >= 0 && *(_BYTE *)(v11 + 140) == 12 )
            {
              v98 = v32;
              v77 = sub_8D4AB0(v11);
              v32 = v98;
              v36 = v77;
            }
            else
            {
              v36 = *(unsigned int *)(v11 + 136);
            }
            sub_15F9450(v32, v36);
            LOWORD(v117[0]) = 257;
            if ( *v10 )
            {
              v116[0] = (__int64)v10;
              LOBYTE(v117[0]) = 3;
            }
            sub_164B780(v114, v116);
            v10 = (const char *)s2;
          }
          sub_12A38F0(a1, a4, v10, v110, v9, 0);
          if ( s2 != v120 )
            j_j___libc_free_0(s2, v120[0] + 1LL);
          goto LABEL_13;
        }
        if ( v12 != 3 )
          sub_127B550("Unsupported ABI variant!", a6, 1);
        if ( sub_127B420(*(_QWORD *)(v8 + 24)) )
        {
          v14 = v11;
          v15 = a1;
          LOWORD(v120[0]) = 259;
          s2 = "tmp";
          v16 = sub_127FE40(a1, v14, (__int64)&s2);
        }
        else
        {
          v15 = a1;
          v22 = sub_127A030(a1[4] + 8LL, *(_QWORD *)(a4 + 120), 0);
          v16 = (_QWORD *)sub_1599EF0(v22);
        }
        v17 = v9;
        v8 += 40;
        ++v9;
        sub_12A38F0(v15, a4, v10, v16, v17, 0);
        a4 = *(_QWORD *)(a4 + 112);
        if ( !a4 )
          goto LABEL_21;
      }
    }
    v12 = *(_DWORD *)(v8 + 12);
    v13 = v12 <= 2;
    if ( v12 != 2 )
      goto LABEL_16;
    goto LABEL_28;
  }
LABEL_21:
  if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
    sub_15E08E0(a3);
  result = *(_QWORD *)(a3 + 88) + 40LL * *(_QWORD *)(a3 + 96);
  if ( v114 != result )
    sub_127B550("Argument mismatch in generation function prolog!", a6, 1);
  return result;
}
