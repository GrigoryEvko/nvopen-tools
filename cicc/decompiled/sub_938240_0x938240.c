// Function: sub_938240
// Address: 0x938240
//
__int64 __fastcall sub_938240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6, char a7)
{
  unsigned __int64 v8; // rcx
  __int64 v9; // r14
  unsigned int v10; // ebx
  char *v11; // r15
  unsigned __int64 v12; // r12
  unsigned int v13; // eax
  bool v14; // cc
  __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 result; // rax
  bool v21; // al
  __int64 v22; // r9
  __int64 v23; // r12
  __int64 v24; // rax
  size_t v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rcx
  int v28; // ecx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  int v31; // r9d
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int *v35; // r15
  unsigned int *v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rax
  char v40; // al
  __int16 v41; // si
  __int64 v42; // rdx
  _BYTE *v43; // rsi
  _QWORD *v44; // r9
  __int64 v45; // r8
  const void *v46; // r14
  __int64 v47; // rbx
  _DWORD *v48; // r12
  size_t v49; // r13
  size_t v50; // r15
  size_t v51; // rdx
  int v52; // eax
  _DWORD *v53; // rsi
  size_t v54; // r10
  size_t v55; // rdx
  int v56; // eax
  __int64 v57; // r10
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  char v60; // al
  __int16 v61; // si
  __int64 v62; // rax
  __int64 v63; // r10
  int v64; // esi
  __int64 v65; // r10
  __int64 v66; // rax
  __int64 v67; // rdx
  unsigned int *v68; // r15
  __int64 v69; // r12
  unsigned int *v70; // rbx
  __int64 v71; // rdx
  __int64 v72; // rsi
  __int64 v73; // rax
  char v74; // al
  __int16 v75; // cx
  __int64 v76; // rax
  int v77; // r9d
  int v78; // ecx
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rax
  unsigned int *v82; // r15
  __int64 v83; // r12
  unsigned int *v84; // rbx
  __int64 v85; // rdx
  __int64 v86; // rsi
  size_t v87; // rax
  __int64 v88; // [rsp+0h] [rbp-110h]
  unsigned int v89; // [rsp+0h] [rbp-110h]
  unsigned int v90; // [rsp+8h] [rbp-108h]
  __int64 v91; // [rsp+8h] [rbp-108h]
  int v92; // [rsp+10h] [rbp-100h]
  unsigned int v93; // [rsp+10h] [rbp-100h]
  int v94; // [rsp+10h] [rbp-100h]
  __int64 v95; // [rsp+10h] [rbp-100h]
  __int64 v96; // [rsp+10h] [rbp-100h]
  __int64 v97; // [rsp+10h] [rbp-100h]
  __int64 v98; // [rsp+10h] [rbp-100h]
  char *v99; // [rsp+10h] [rbp-100h]
  __int64 v100; // [rsp+10h] [rbp-100h]
  unsigned int v101; // [rsp+10h] [rbp-100h]
  int v102; // [rsp+10h] [rbp-100h]
  char *v103; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v104; // [rsp+18h] [rbp-F8h]
  size_t v105; // [rsp+18h] [rbp-F8h]
  __int64 v106; // [rsp+18h] [rbp-F8h]
  __int16 v107; // [rsp+20h] [rbp-F0h]
  __int16 v108; // [rsp+22h] [rbp-EEh]
  __int16 v109; // [rsp+24h] [rbp-ECh]
  char *v110; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v111; // [rsp+28h] [rbp-E8h]
  int v112; // [rsp+30h] [rbp-E0h]
  __int64 v113; // [rsp+30h] [rbp-E0h]
  __int64 v114; // [rsp+30h] [rbp-E0h]
  char *v115; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v117; // [rsp+40h] [rbp-D0h]
  _QWORD *v118; // [rsp+40h] [rbp-D0h]
  __int16 v119; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v121; // [rsp+58h] [rbp-B8h]
  __int64 v122[2]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v123[2]; // [rsp+70h] [rbp-A0h] BYREF
  char *v124; // [rsp+80h] [rbp-90h] BYREF
  __int64 v125; // [rsp+88h] [rbp-88h]
  _QWORD v126[2]; // [rsp+90h] [rbp-80h] BYREF
  __int16 v127; // [rsp+A0h] [rbp-70h]
  void *s2; // [rsp+B0h] [rbp-60h] BYREF
  size_t v129; // [rsp+B8h] [rbp-58h]
  _QWORD v130[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v131; // [rsp+D0h] [rbp-40h]

  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    sub_B2C6D0(a3);
  v121 = *(_QWORD *)(a3 + 96);
  if ( sub_938130(*(_QWORD *)(a1 + 32), a2) )
  {
    s2 = "agg.result";
    v131 = 259;
    sub_BD6B50(v121, &s2);
    v121 += 40LL;
  }
  v9 = *(_QWORD *)(a2 + 16) + 40LL;
  if ( a4 )
  {
    v10 = 1;
    while ( 1 )
    {
      v11 = *(char **)(a4 + 8);
      v12 = *(_QWORD *)(v9 + 24);
      if ( v11 )
        break;
      v11 = "temp_param";
      if ( (*(_BYTE *)(a4 + 172) & 1) != 0 )
        v11 = "this";
      v13 = *(_DWORD *)(v9 + 12);
      v14 = v13 <= 2;
      if ( v13 == 2 )
      {
LABEL_28:
        if ( !a7 || !*(_BYTE *)(v9 + 16) )
          goto LABEL_32;
        if ( !unk_4D046B0 )
          goto LABEL_31;
        v43 = (_BYTE *)sub_BD5D20(a3);
        if ( v43 )
        {
          s2 = v130;
          sub_9373F0((__int64 *)&s2, v43, (__int64)&v43[v42]);
          v44 = s2;
          v45 = *(_QWORD *)&dword_4D04688[2];
          if ( !*(_QWORD *)&dword_4D04688[2] )
            goto LABEL_109;
        }
        else
        {
          v44 = v130;
          v129 = 0;
          v45 = *(_QWORD *)&dword_4D04688[2];
          s2 = v130;
          LOBYTE(v130[0]) = 0;
          if ( !*(_QWORD *)&dword_4D04688[2] )
            goto LABEL_32;
        }
        v104 = v12;
        v95 = v9;
        v46 = v44;
        v90 = v10;
        v47 = v45;
        v88 = a4;
        v48 = dword_4D04688;
        v49 = v129;
        v110 = v11;
        do
        {
          while ( 1 )
          {
            v50 = *(_QWORD *)(v47 + 40);
            v51 = v49;
            if ( v50 <= v49 )
              v51 = *(_QWORD *)(v47 + 40);
            if ( v51 )
            {
              v52 = memcmp(*(const void **)(v47 + 32), v46, v51);
              if ( v52 )
                break;
            }
            if ( (__int64)(v50 - v49) >= 0x80000000LL )
              goto LABEL_66;
            if ( (__int64)(v50 - v49) > (__int64)0xFFFFFFFF7FFFFFFFLL )
            {
              v52 = v50 - v49;
              break;
            }
LABEL_57:
            v47 = *(_QWORD *)(v47 + 24);
            if ( !v47 )
              goto LABEL_67;
          }
          if ( v52 < 0 )
            goto LABEL_57;
LABEL_66:
          v48 = (_DWORD *)v47;
          v47 = *(_QWORD *)(v47 + 16);
        }
        while ( v47 );
LABEL_67:
        v53 = v48;
        v54 = v49;
        v44 = v46;
        v11 = v110;
        v10 = v90;
        v12 = v104;
        v9 = v95;
        a4 = v88;
        if ( v53 == dword_4D04688 )
          goto LABEL_109;
        v8 = *((_QWORD *)v53 + 5);
        v55 = v54;
        if ( v8 <= v54 )
          v55 = *((_QWORD *)v53 + 5);
        if ( v55 )
        {
          v105 = v54;
          v111 = *((_QWORD *)v53 + 5);
          v118 = v44;
          v56 = memcmp(v44, *((const void **)v53 + 4), v55);
          v44 = v118;
          v8 = v111;
          v54 = v105;
          if ( v56 )
          {
LABEL_75:
            if ( v56 < 0 )
              goto LABEL_109;
LABEL_76:
            if ( v44 != v130 )
              j_j___libc_free_0(v44, v130[0] + 1LL);
LABEL_31:
            if ( *(_BYTE *)(v9 + 33) )
              goto LABEL_32;
            v112 = unk_4D0463C;
            if ( unk_4D0463C )
              v112 = sub_90AA40(*(_QWORD *)(a1 + 32), v121);
            if ( *(char *)(v12 + 142) >= 0 && *(_BYTE *)(v12 + 140) == 12 )
              v58 = (unsigned int)sub_8D4AB0(v12);
            else
              v58 = *(unsigned int *)(v12 + 136);
            v119 = 510;
            if ( v58 )
            {
              _BitScanReverse64(&v58, v58);
              v119 = (2 * (63 - (v58 ^ 0x3F))) & 0x1FE;
            }
            s2 = "tmp";
            v131 = 259;
            v23 = sub_921D70(a1, v12, (__int64)&s2, v8);
            v127 = 257;
            v96 = *(_QWORD *)(v23 + 8);
            v59 = sub_AA4E30(*(_QWORD *)(a1 + 96));
            v60 = sub_AE5020(v59, v96);
            HIBYTE(v61) = HIBYTE(v108);
            v131 = 257;
            LOBYTE(v61) = v60;
            v108 = v61;
            v62 = sub_BD2C40(80, unk_3F10A14);
            v63 = v62;
            if ( v62 )
            {
              v64 = v96;
              v97 = v62;
              sub_B4D190(v62, v64, v121, (unsigned int)&s2, v112, (unsigned __int8)v108, 0, 0);
              v63 = v97;
            }
            v98 = v63;
            (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
              *(_QWORD *)(a1 + 136),
              v63,
              &v124,
              *(_QWORD *)(a1 + 104),
              *(_QWORD *)(a1 + 112));
            v65 = v98;
            v66 = *(_QWORD *)(a1 + 48);
            v67 = 16LL * *(unsigned int *)(a1 + 56);
            if ( v66 != v66 + v67 )
            {
              v99 = v11;
              v68 = (unsigned int *)(v66 + v67);
              v91 = v23;
              v69 = v65;
              v89 = v10;
              v70 = *(unsigned int **)(a1 + 48);
              do
              {
                v71 = *((_QWORD *)v70 + 1);
                v72 = *v70;
                v70 += 4;
                sub_B99FD0(v69, v72, v71);
              }
              while ( v68 != v70 );
              v65 = v69;
              v11 = v99;
              v10 = v89;
              v23 = v91;
            }
            v100 = v65;
            *(_WORD *)(v65 + 2) = v119 | *(_WORD *)(v65 + 2) & 0xFF81;
            v73 = sub_AA4E30(*(_QWORD *)(a1 + 96));
            v74 = sub_AE5020(v73, *(_QWORD *)(v100 + 8));
            HIBYTE(v75) = HIBYTE(v107);
            v131 = 257;
            LOBYTE(v75) = v74;
            v107 = v75;
            v76 = sub_BD2C40(80, unk_3F10A10);
            if ( v76 )
            {
              v78 = v112;
              v113 = v76;
              sub_B4D3C0(v76, v100, v23, v78, (unsigned __int8)v107, v77, 0, 0);
              v76 = v113;
            }
            v114 = v76;
            (*(void (__fastcall **)(_QWORD, __int64, void **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
              *(_QWORD *)(a1 + 136),
              v76,
              &s2,
              *(_QWORD *)(a1 + 104),
              *(_QWORD *)(a1 + 112));
            v79 = *(_QWORD *)(a1 + 48);
            v80 = 16LL * *(unsigned int *)(a1 + 56);
            v81 = v114;
            if ( v79 != v79 + v80 )
            {
              v115 = v11;
              v82 = (unsigned int *)(v79 + v80);
              v106 = v23;
              v83 = v81;
              v101 = v10;
              v84 = *(unsigned int **)(a1 + 48);
              do
              {
                v85 = *((_QWORD *)v84 + 1);
                v86 = *v84;
                v84 += 4;
                sub_B99FD0(v83, v86, v85);
              }
              while ( v82 != v84 );
              v81 = v83;
              v11 = v115;
              v10 = v101;
              v23 = v106;
            }
            *(_WORD *)(v81 + 2) = v119 | *(_WORD *)(v81 + 2) & 0xFF81;
            v122[0] = (__int64)v123;
            sub_9373F0(v122, "__val_param", (__int64)"");
            v87 = strlen(v11);
            if ( v87 <= 0x3FFFFFFFFFFFFFFFLL - v122[1] )
            {
              sub_2241490(v122, v11, v87, v122);
              v131 = 260;
              s2 = v122;
              sub_BD6B50(v121, &s2);
              if ( (_QWORD *)v122[0] != v123 )
                j_j___libc_free_0(v122[0], v123[0] + 1LL);
              v22 = *(unsigned __int8 *)(v9 + 33);
              goto LABEL_34;
            }
LABEL_112:
            sub_4262D8((__int64)"basic_string::append");
          }
        }
        v57 = v54 - v8;
        if ( v57 >= 0x80000000LL )
          goto LABEL_76;
        if ( v57 >= (__int64)0xFFFFFFFF80000000LL )
        {
          v56 = v57;
          goto LABEL_75;
        }
LABEL_109:
        if ( v44 != v130 )
          j_j___libc_free_0(v44, v130[0] + 1LL);
LABEL_32:
        v21 = sub_91B770(v12);
        v22 = *(unsigned __int8 *)(v9 + 33);
        v23 = v121;
        if ( !v21 && !(_BYTE)v22 )
          sub_91B8A0("Non-aggregate arguments passed indirectly are not supported!", a6, 1);
LABEL_34:
        sub_9446C0(a1, a4, v11, v23, v10, v22);
LABEL_13:
        a4 = *(_QWORD *)(a4 + 112);
        v121 += 40LL;
        v9 += 40;
        ++v10;
        if ( !a4 )
          goto LABEL_21;
      }
      else
      {
LABEL_16:
        if ( v14 )
        {
          if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
            sub_B2C6D0(a3);
          if ( v121 == *(_QWORD *)(a3 + 96) + 40LL * *(_QWORD *)(a3 + 104) )
            sub_91B8A0("Argument mismatch in generation function prolog!", a6, 1);
          LOBYTE(v126[0]) = 0;
          v124 = (char *)v126;
          v125 = 0;
          v117 = v121;
          if ( !sub_91B770(v12) )
            goto LABEL_11;
          v25 = strlen(v11);
          sub_2241130(&v124, 0, v125, v11, v25);
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v125) <= 4 )
            goto LABEL_112;
          sub_2241490(&v124, ".addr", 5, v26);
          s2 = "tmp";
          v131 = 259;
          v117 = sub_921D70(a1, v12, (__int64)&s2, v27);
          v28 = unk_4D0463C;
          if ( unk_4D0463C )
          {
            v28 = sub_90AA40(*(_QWORD *)(a1 + 32), v117);
            if ( *(char *)(v12 + 142) < 0 )
              goto LABEL_41;
LABEL_40:
            if ( *(_BYTE *)(v12 + 140) != 12 )
              goto LABEL_41;
            v102 = v28;
            LODWORD(v29) = sub_8D4AB0(v12);
            v28 = v102;
            v29 = (unsigned int)v29;
          }
          else
          {
            if ( *(char *)(v12 + 142) >= 0 )
              goto LABEL_40;
LABEL_41:
            v29 = *(unsigned int *)(v12 + 136);
          }
          if ( v29 )
          {
            _BitScanReverse64(&v29, v29);
            BYTE1(v29) = HIBYTE(v109);
            LOBYTE(v29) = 63 - (v29 ^ 0x3F);
            v109 = v29;
          }
          else
          {
            v94 = v28;
            v39 = sub_AA4E30(*(_QWORD *)(a1 + 96));
            v40 = sub_AE5020(v39, *(_QWORD *)(v121 + 8));
            HIBYTE(v41) = HIBYTE(v109);
            v28 = v94;
            LOBYTE(v41) = v40;
            v109 = v41;
          }
          v92 = v28;
          v131 = 257;
          v30 = sub_BD2C40(80, unk_3F10A10);
          v32 = v30;
          if ( v30 )
            sub_B4D3C0(v30, v121, v117, v92, (unsigned __int8)v109, v31, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, void **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
            *(_QWORD *)(a1 + 136),
            v32,
            &s2,
            *(_QWORD *)(a1 + 104),
            *(_QWORD *)(a1 + 112));
          v33 = *(_QWORD *)(a1 + 48);
          v34 = 16LL * *(unsigned int *)(a1 + 56);
          if ( v33 != v33 + v34 )
          {
            v103 = v11;
            v35 = (unsigned int *)(v33 + v34);
            v93 = v10;
            v36 = *(unsigned int **)(a1 + 48);
            do
            {
              v37 = *((_QWORD *)v36 + 1);
              v38 = *v36;
              v36 += 4;
              sub_B99FD0(v32, v38, v37);
            }
            while ( v35 != v36 );
            v11 = v103;
            v10 = v93;
          }
          v131 = 257;
          if ( *v11 )
          {
            s2 = v11;
            LOBYTE(v131) = 3;
          }
          sub_BD6B50(v121, &s2);
          v11 = v124;
LABEL_11:
          sub_9446C0(a1, a4, v11, v117, v10, 0);
          if ( v124 != (char *)v126 )
            j_j___libc_free_0(v124, v126[0] + 1LL);
          goto LABEL_13;
        }
        if ( v13 != 3 )
          sub_91B8A0("Unsupported ABI variant!", a6, 1);
        if ( sub_91B770(*(_QWORD *)(v9 + 24)) )
        {
          v16 = v12;
          v17 = a1;
          v131 = 259;
          s2 = "tmp";
          v18 = sub_921D70(a1, v16, (__int64)&s2, v15);
        }
        else
        {
          v17 = a1;
          v24 = sub_91A390(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a4 + 120), 0, v15);
          v18 = sub_ACA8A0(v24);
        }
        v19 = v10;
        v9 += 40;
        ++v10;
        sub_9446C0(v17, a4, v11, v18, v19, 0);
        a4 = *(_QWORD *)(a4 + 112);
        if ( !a4 )
          goto LABEL_21;
      }
    }
    v13 = *(_DWORD *)(v9 + 12);
    v14 = v13 <= 2;
    if ( v13 != 2 )
      goto LABEL_16;
    goto LABEL_28;
  }
LABEL_21:
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    sub_B2C6D0(a3);
  result = *(_QWORD *)(a3 + 96) + 40LL * *(_QWORD *)(a3 + 104);
  if ( v121 != result )
    sub_91B8A0("Argument mismatch in generation function prolog!", a6, 1);
  return result;
}
