// Function: sub_2CD3FA0
// Address: 0x2cd3fa0
//
_QWORD *__fastcall sub_2CD3FA0(_QWORD *a1, char *a2, __int64 a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *v11; // r12
  _QWORD *v12; // r12
  _QWORD *v13; // rbx
  char v15; // al
  char v16; // al
  char v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rdi
  char v20; // r13
  _QWORD *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 **v24; // r15
  __int64 v25; // rax
  __int64 **v26; // rax
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  char v29; // al
  __int16 v30; // dx
  _QWORD *v31; // rax
  __int64 v32; // r9
  __int64 v33; // r14
  __int64 *v34; // r13
  __int64 *v35; // r15
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 **v38; // r15
  unsigned __int64 v39; // rax
  __int64 *v40; // rsi
  __int64 *v41; // rdi
  _QWORD *v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 **v45; // r15
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  int v51; // edi
  char v52; // si
  int v53; // esi
  __int64 v54; // rax
  char v55; // al
  __int16 v56; // dx
  _QWORD *v57; // rax
  _QWORD *v58; // r10
  __int64 v59; // rdx
  __int64 *v60; // r15
  __int64 v61; // r10
  __int64 v62; // rdx
  unsigned int v63; // esi
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 *v66; // rsi
  unsigned __int64 v67; // r10
  __int64 *v68; // r12
  char v69; // r13
  __int64 v70; // rdi
  __int64 v71; // rdi
  char v72; // r13
  const char *v73; // rdx
  __int64 v74; // rsi
  unsigned __int8 *v75; // rsi
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  int v78; // r9d
  __int64 v79; // r12
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // rbx
  __int64 v83; // r13
  int v84; // eax
  _BYTE *v85; // rax
  __int64 v86; // r15
  __int64 *v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rsi
  int v90; // eax
  __int64 *v91; // r9
  _QWORD *v92; // rsi
  char v93; // al
  __int64 v94; // rax
  _QWORD *i; // r13
  __int64 v98; // [rsp+10h] [rbp-180h]
  __int64 v99; // [rsp+18h] [rbp-178h]
  __int64 *v100; // [rsp+20h] [rbp-170h]
  unsigned __int64 v101; // [rsp+28h] [rbp-168h]
  unsigned __int64 v102; // [rsp+28h] [rbp-168h]
  _QWORD *v103; // [rsp+28h] [rbp-168h]
  __int64 v104; // [rsp+28h] [rbp-168h]
  unsigned __int64 v105; // [rsp+28h] [rbp-168h]
  unsigned __int64 v106; // [rsp+28h] [rbp-168h]
  char v107; // [rsp+35h] [rbp-15Bh]
  __int16 v108; // [rsp+36h] [rbp-15Ah]
  char v109; // [rsp+36h] [rbp-15Ah]
  __int16 v110; // [rsp+38h] [rbp-158h]
  __int64 v111; // [rsp+40h] [rbp-150h]
  __int64 v112; // [rsp+40h] [rbp-150h]
  __int64 v113; // [rsp+48h] [rbp-148h]
  __int64 *v114; // [rsp+48h] [rbp-148h]
  __int64 *v115; // [rsp+48h] [rbp-148h]
  __int64 v116; // [rsp+50h] [rbp-140h]
  __int64 v117; // [rsp+50h] [rbp-140h]
  __int64 v118; // [rsp+58h] [rbp-138h]
  char v119; // [rsp+67h] [rbp-129h] BYREF
  int v120; // [rsp+68h] [rbp-128h]
  __int64 *v121[2]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v122[2]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v123; // [rsp+90h] [rbp-100h]
  __int64 *v124; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v125; // [rsp+A8h] [rbp-E8h]
  __int64 v126[2]; // [rsp+B0h] [rbp-E0h] BYREF
  __int16 v127; // [rsp+C0h] [rbp-D0h]
  __int64 *v128; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v129; // [rsp+D8h] [rbp-B8h]
  char v130; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v131; // [rsp+100h] [rbp-90h]
  __int64 v132; // [rsp+108h] [rbp-88h]
  __int64 v133; // [rsp+110h] [rbp-80h]
  __int64 v134; // [rsp+128h] [rbp-68h]
  void *v135; // [rsp+150h] [rbp-40h]

  v4 = a3 + 24;
  v5 = *(_QWORD *)(a3 + 32);
  v6 = *a2;
  v119 = 0;
  v107 = v6;
  v98 = v4;
  v99 = v5;
  if ( v4 == v5 )
  {
    v12 = a1 + 4;
    v13 = a1 + 10;
    goto LABEL_11;
  }
  do
  {
    v7 = *(_QWORD *)(v99 + 24);
    v111 = v99 + 16;
    v99 = *(_QWORD *)(v99 + 8);
    v118 = v7;
    if ( v111 != v7 )
    {
      while ( 1 )
      {
        v8 = v118 + 24;
        v9 = *(_QWORD *)(v118 + 32);
        v118 = *(_QWORD *)(v118 + 8);
        v113 = v8;
        v116 = v9;
        if ( v8 != v9 )
          break;
LABEL_7:
        if ( v111 == v118 )
          goto LABEL_8;
      }
      while ( 1 )
      {
        v10 = v116;
        v11 = (_QWORD *)(v116 - 24);
        v116 = *(_QWORD *)(v116 + 8);
        switch ( *(_BYTE *)v11 )
        {
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2C:
          case 0x2E:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x54:
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x5E:
          case 0x5F:
          case 0x60:
            goto LABEL_6;
          case 0x2B:
            sub_2CD2800(&v119, (__int64)v11, (__int64)"__nv_add_fp128", 0xEu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x2D:
            sub_2CD2800(&v119, (__int64)v11, (__int64)"__nv_sub_fp128", 0xEu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x2F:
            sub_2CD2800(&v119, (__int64)v11, (__int64)"__nv_mul_fp128", 0xEu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x30:
            sub_2CD3C90(&v119, v11, (__int64)"__nv_udiv128", 0xCu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x31:
            sub_2CD3C90(&v119, v11, (__int64)"__nv_idiv128", 0xCu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x32:
            sub_2CD2800(&v119, (__int64)v11, (__int64)"__nv_div_fp128", 0xEu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x33:
            sub_2CD3C90(&v119, v11, (__int64)"__nv_urem128", 0xCu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x34:
            sub_2CD3C90(&v119, v11, (__int64)"__nv_irem128", 0xCu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x35:
            sub_2CD2800(&v119, (__int64)v11, (__int64)"__nv_rem_fp128", 0xEu);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x3D:
            if ( !sub_BCAC40(*(_QWORD *)(v10 - 16), 128) && *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL) != 5 )
              goto LABEL_6;
            sub_23D0AB0((__int64)&v128, (__int64)v11, 0, 0, 0);
            v42 = (_QWORD *)sub_BD5C60((__int64)v11);
            v43 = (__int64 *)sub_BCB2E0(v42);
            v44 = sub_BCDA70(v43, 2);
            v127 = 257;
            v45 = (__int64 **)v44;
            v46 = *(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL);
            if ( (unsigned int)*(unsigned __int8 *)(v46 + 8) - 17 <= 1 )
              v46 = **(_QWORD **)(v46 + 16);
            v47 = sub_BCE770(v45, *(_DWORD *)(v46 + 8) >> 8);
            v48 = *(_QWORD *)(v10 - 56);
            v49 = v47;
            v50 = *(_QWORD *)(v48 + 8);
            if ( v49 == v50 )
              goto LABEL_73;
            v51 = *(unsigned __int8 *)(v50 + 8);
            v52 = *(_BYTE *)(v50 + 8);
            if ( (unsigned int)(v51 - 17) > 1 )
            {
              if ( (_BYTE)v51 != 14 )
                goto LABEL_126;
            }
            else if ( *(_BYTE *)(**(_QWORD **)(v50 + 16) + 8LL) != 14 )
            {
              goto LABEL_66;
            }
            v78 = *(unsigned __int8 *)(v49 + 8);
            if ( (unsigned int)(v78 - 17) <= 1 )
              LOBYTE(v78) = *(_BYTE *)(**(_QWORD **)(v49 + 16) + 8LL);
            if ( (_BYTE)v78 == 12 )
            {
              v48 = sub_2CD24F0((__int64 *)&v128, 0x2Fu, v48, (__int64 **)v49, (__int64)&v124, 0, (int)v121[0], 0);
              goto LABEL_73;
            }
LABEL_66:
            if ( v51 == 18 )
              goto LABEL_67;
LABEL_126:
            if ( v51 == 17 )
LABEL_67:
              v52 = *(_BYTE *)(**(_QWORD **)(v50 + 16) + 8LL);
            if ( v52 != 12 )
              goto LABEL_72;
            v53 = *(unsigned __int8 *)(v49 + 8);
            if ( (unsigned int)(v53 - 17) <= 1 )
              LOBYTE(v53) = *(_BYTE *)(**(_QWORD **)(v49 + 16) + 8LL);
            if ( (_BYTE)v53 == 14 )
              v48 = sub_2CD24F0((__int64 *)&v128, 0x30u, v48, (__int64 **)v49, (__int64)&v124, 0, (int)v121[0], 0);
            else
LABEL_72:
              v48 = sub_2CD24F0((__int64 *)&v128, 0x31u, v48, (__int64 **)v49, (__int64)&v124, 0, (int)v121[0], 0);
LABEL_73:
            v102 = v48;
            v123 = 257;
            v54 = sub_AA4E30(v131);
            v55 = sub_AE5020(v54, (__int64)v45);
            HIBYTE(v56) = HIBYTE(v108);
            LOBYTE(v56) = v55;
            v127 = 257;
            v108 = v56;
            v57 = sub_BD2C40(80, 1u);
            v58 = v57;
            if ( v57 )
            {
              v59 = v102;
              v103 = v57;
              sub_B4D190((__int64)v57, (__int64)v45, v59, (__int64)&v124, 0, v108, 0, 0);
              v58 = v103;
            }
            v104 = (__int64)v58;
            (*(void (__fastcall **)(__int64, _QWORD *, __int64 **, __int64, __int64))(*(_QWORD *)v134 + 16LL))(
              v134,
              v58,
              v121,
              v132,
              v133);
            v60 = v128;
            v61 = v104;
            if ( v128 != &v128[2 * (unsigned int)v129] )
            {
              v100 = &v128[2 * (unsigned int)v129];
              do
              {
                v62 = v60[1];
                v63 = *(_DWORD *)v60;
                v60 += 2;
                sub_B99FD0(v104, v63, v62);
              }
              while ( v100 != v60 );
              v61 = v104;
            }
            v122[0] = (__int64)v11;
            v105 = v61;
            v121[0] = v122;
            v121[1] = (__int64 *)0x200000001LL;
            sub_9B8FE0(v61, v122, 1);
            _BitScanReverse64(&v64, 1LL << (*(_WORD *)(v10 - 22) >> 1));
            *(_WORD *)(v105 + 2) = *(_WORD *)(v105 + 2) & 0xFF81 | (2 * (63 - (v64 ^ 0x3F)));
            v127 = 257;
            v65 = sub_2CD24F0((__int64 *)&v128, 0x31u, v105, *(__int64 ***)(v10 - 16), (__int64)&v124, 0, v120, 0);
            sub_BD84D0((__int64)v11, v65);
            v66 = *(__int64 **)(v10 + 24);
            v67 = v105;
            v124 = v66;
            v68 = (__int64 *)(v105 + 48);
            if ( !v66 )
            {
              if ( v68 == (__int64 *)&v124 )
                goto LABEL_83;
              v76 = *(_QWORD *)(v105 + 48);
              if ( !v76 )
                goto LABEL_83;
LABEL_135:
              v106 = v67;
              sub_B91220((__int64)v68, v76);
              v67 = v106;
              goto LABEL_136;
            }
            sub_B96E90((__int64)&v124, (__int64)v66, 1);
            if ( v68 == (__int64 *)&v124 )
            {
              if ( v124 )
                sub_B91220((__int64)&v124, (__int64)v124);
              goto LABEL_83;
            }
            v67 = v105;
            v76 = *(_QWORD *)(v105 + 48);
            if ( v76 )
              goto LABEL_135;
LABEL_136:
            v77 = (unsigned __int8 *)v124;
            *(_QWORD *)(v67 + 48) = v124;
            if ( v77 )
              sub_B976B0((__int64)&v124, v77, (__int64)v68);
LABEL_83:
            v41 = v121[0];
            v119 = 1;
            if ( v121[0] != v122 )
LABEL_55:
              _libc_free((unsigned __int64)v41);
LABEL_56:
            nullsub_61();
            v135 = &unk_49DA100;
            nullsub_63();
            if ( v128 == (__int64 *)&v130 )
              goto LABEL_6;
            _libc_free((unsigned __int64)v128);
            if ( v113 == v116 )
              goto LABEL_7;
            continue;
          case 0x3E:
            if ( !sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 88) + 8LL), 128)
              && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 88) + 8LL) + 8LL) != 5 )
            {
              goto LABEL_6;
            }
            sub_23D0AB0((__int64)&v128, (__int64)v11, 0, 0, 0);
            v21 = (_QWORD *)sub_BD5C60((__int64)v11);
            v22 = (__int64 *)sub_BCB2E0(v21);
            v23 = sub_BCDA70(v22, 2);
            v127 = 257;
            v24 = (__int64 **)v23;
            v25 = *(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL);
            if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
              v25 = **(_QWORD **)(v25 + 16);
            v26 = (__int64 **)sub_BCE770(v24, *(_DWORD *)(v25 + 8) >> 8);
            v101 = sub_2CD24F0((__int64 *)&v128, 0x31u, *(_QWORD *)(v10 - 56), v26, (__int64)&v124, 0, (int)v121[0], 0);
            v127 = 257;
            v27 = sub_2CD24F0((__int64 *)&v128, 0x31u, *(_QWORD *)(v10 - 88), v24, (__int64)&v124, 0, (int)v121[0], 0);
            v28 = sub_AA4E30(v131);
            v29 = sub_AE5020(v28, *(_QWORD *)(v27 + 8));
            HIBYTE(v30) = HIBYTE(v110);
            v127 = 257;
            LOBYTE(v30) = v29;
            v110 = v30;
            v31 = sub_BD2C40(80, unk_3F10A10);
            v33 = (__int64)v31;
            if ( v31 )
              sub_B4D3C0((__int64)v31, v27, v101, 0, v110, v32, 0, 0);
            (*(void (__fastcall **)(__int64, __int64, __int64 **, __int64, __int64))(*(_QWORD *)v134 + 16LL))(
              v134,
              v33,
              &v124,
              v132,
              v133);
            v34 = v128;
            v35 = &v128[2 * (unsigned int)v129];
            if ( v128 != v35 )
            {
              do
              {
                v36 = v34[1];
                v37 = *(_DWORD *)v34;
                v34 += 2;
                sub_B99FD0(v33, v37, v36);
              }
              while ( v35 != v34 );
            }
            v38 = (__int64 **)(v33 + 48);
            v124 = v126;
            v126[0] = (__int64)v11;
            v125 = 0x200000001LL;
            sub_9B8FE0(v33, v126, 1);
            _BitScanReverse64(&v39, 1LL << (*(_WORD *)(v10 - 22) >> 1));
            *(_WORD *)(v33 + 2) = *(_WORD *)(v33 + 2) & 0xFF81 | (2 * (63 - (v39 ^ 0x3F)));
            v40 = *(__int64 **)(v10 + 24);
            v121[0] = v40;
            if ( !v40 )
            {
              if ( v38 == v121 )
                goto LABEL_54;
              v74 = *(_QWORD *)(v33 + 48);
              if ( !v74 )
                goto LABEL_54;
LABEL_130:
              sub_B91220(v33 + 48, v74);
              goto LABEL_131;
            }
            sub_B96E90((__int64)v121, (__int64)v40, 1);
            if ( v38 == v121 )
            {
              if ( v121[0] )
                sub_B91220(v33 + 48, (__int64)v121[0]);
              goto LABEL_54;
            }
            v74 = *(_QWORD *)(v33 + 48);
            if ( v74 )
              goto LABEL_130;
LABEL_131:
            v75 = (unsigned __int8 *)v121[0];
            *(__int64 **)(v33 + 48) = v121[0];
            if ( v75 )
              sub_B976B0((__int64)v121, v75, v33 + 48);
LABEL_54:
            sub_B43D60(v11);
            v41 = v124;
            v119 = 1;
            if ( v124 != v126 )
              goto LABEL_55;
            goto LABEL_56;
          case 0x46:
            v19 = *(_QWORD *)(v10 - 16);
            v20 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL) + 8LL);
            if ( v20 == 5 )
            {
              if ( sub_BCAC40(v19, 8) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_uint8", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 16) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_uint16", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 32) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_uint32", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( !sub_BCAC40(*(_QWORD *)(v10 - 16), 64) )
                {
                  if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 128) )
                    sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_uint128", 0x15u);
                  goto LABEL_6;
                }
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_uint64", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            else if ( (unsigned int)sub_BCB060(v19) == 128 )
            {
              if ( v20 == 2 )
              {
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_f32_u128_rz", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( v20 != 3 )
                  goto LABEL_6;
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_f64_u128_rz", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            else
            {
LABEL_6:
              if ( v113 == v116 )
                goto LABEL_7;
            }
            break;
          case 0x47:
            v71 = *(_QWORD *)(v10 - 16);
            v72 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL) + 8LL);
            if ( v72 == 5 )
            {
              if ( sub_BCAC40(v71, 8) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_int8", 0x12u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 16) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_int16", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 32) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_int32", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( !sub_BCAC40(*(_QWORD *)(v10 - 16), 64) )
                {
                  if ( sub_BCAC40(*(_QWORD *)(v10 - 16), 128) )
                    sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_int128", 0x14u);
                  goto LABEL_6;
                }
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_int64", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            else
            {
              if ( (unsigned int)sub_BCB060(v71) != 128 )
                goto LABEL_6;
              if ( v72 == 2 )
              {
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_f32_i128_rz", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( v72 != 3 )
                  goto LABEL_6;
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_f64_i128_rz", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            continue;
          case 0x48:
            v69 = *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL);
            v70 = *(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL);
            if ( v69 == 5 )
            {
              if ( sub_BCAC40(v70, 8) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_uint8_to_fp128", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 16) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_uint16_to_fp128", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 32) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_uint32_to_fp128", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 64) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_uint64_to_fp128", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( !sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 128) )
                  goto LABEL_6;
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_uint128_to_fp128", 0x15u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            else
            {
              if ( (unsigned int)sub_BCB060(v70) != 128 )
                goto LABEL_6;
              if ( v69 == 2 )
              {
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_u128_f32_rn", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( v69 != 3 )
                  goto LABEL_6;
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_u128_f64_rn", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            continue;
          case 0x49:
            v17 = *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL);
            v18 = *(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL);
            if ( v17 == 5 )
            {
              if ( sub_BCAC40(v18, 8) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_int8_to_fp128", 0x12u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 16) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_int16_to_fp128", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 32) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_int32_to_fp128", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else if ( sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 64) )
              {
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_int64_to_fp128", 0x13u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( !sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL), 128) )
                  goto LABEL_6;
                sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_int128_to_fp128", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            else
            {
              if ( (unsigned int)sub_BCB060(v18) != 128 )
                goto LABEL_6;
              if ( v17 == 2 )
              {
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_i128_f32_rn", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
              else
              {
                if ( v17 != 3 )
                  goto LABEL_6;
                sub_2CD3E30(&v119, v11, (__int64)"__nv_cvt_i128_f64_rn", 0x14u);
                if ( v113 == v116 )
                  goto LABEL_7;
              }
            }
            continue;
          case 0x4A:
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL) + 8LL) != 5 )
              goto LABEL_6;
            v16 = *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL);
            if ( v16 == 2 )
            {
              sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_float", 0x13u);
              if ( v113 == v116 )
                goto LABEL_7;
            }
            else
            {
              if ( v16 != 3 )
                goto LABEL_6;
              sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_fp128_to_double", 0x14u);
              if ( v113 == v116 )
                goto LABEL_7;
            }
            continue;
          case 0x4B:
            if ( *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL) != 5 )
              goto LABEL_6;
            v15 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 56) + 8LL) + 8LL);
            if ( v15 == 2 )
            {
              sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_float_to_fp128", 0x13u);
              if ( v113 == v116 )
                goto LABEL_7;
            }
            else
            {
              if ( v15 != 3 )
                goto LABEL_6;
              sub_2CD2C00(&v119, (__int64)v11, (__int64)"__nv_double_to_fp128", 0x14u);
              if ( v113 == v116 )
                goto LABEL_7;
            }
            continue;
          case 0x53:
            switch ( *(_WORD *)(v10 - 22) & 0x3F )
            {
              case 1:
                v73 = "__nv_fcmp_oeq";
                goto LABEL_110;
              case 2:
                v73 = "__nv_fcmp_ogt";
                goto LABEL_110;
              case 3:
                v73 = "__nv_fcmp_oge";
                goto LABEL_110;
              case 4:
                v73 = "__nv_fcmp_olt";
                goto LABEL_110;
              case 5:
                v73 = "__nv_fcmp_ole";
                goto LABEL_110;
              case 6:
                v73 = "__nv_fcmp_one";
                goto LABEL_110;
              case 7:
                v73 = "__nv_fcmp_ord";
                goto LABEL_110;
              case 8:
                v73 = "__nv_fcmp_uno";
                goto LABEL_110;
              case 9:
                v73 = "__nv_fcmp_ueq";
                goto LABEL_110;
              case 0xA:
                v73 = "__nv_fcmp_ugt";
                goto LABEL_110;
              case 0xB:
                v73 = "__nv_fcmp_uge";
                goto LABEL_110;
              case 0xC:
                v73 = "__nv_fcmp_ult";
                goto LABEL_110;
              case 0xD:
                v73 = "__nv_fcmp_ule";
                goto LABEL_110;
              case 0xE:
                v73 = "__nv_fcmp_une";
LABEL_110:
                sub_2CD2800(&v119, (__int64)v11, (__int64)v73, 0xDu);
                if ( v113 == v116 )
                  goto LABEL_7;
                continue;
              default:
                goto LABEL_6;
            }
          case 0x55:
            sub_2CD3350(&v119, (__int64)v11);
            goto LABEL_6;
          default:
            goto LABEL_255;
        }
      }
    }
LABEL_8:
    ;
  }
  while ( v98 != v99 );
  v109 = v119;
  if ( v107 )
  {
    LOBYTE(v124) = 0;
    v112 = *(_QWORD *)(a3 + 32);
    if ( v98 != v112 )
    {
      while ( 1 )
      {
        v79 = *(_QWORD *)(v112 + 24);
        v117 = v112 + 16;
        v112 = *(_QWORD *)(v112 + 8);
        if ( v117 != v79 )
          break;
LABEL_239:
        if ( v98 == v112 )
        {
          v109 |= (unsigned __int8)v124;
          goto LABEL_10;
        }
      }
      while ( 1 )
      {
        v80 = v79;
        v79 = *(_QWORD *)(v79 + 8);
        v81 = *(_QWORD *)(v80 + 32);
        v82 = v80 + 24;
        if ( v80 + 24 != v81 )
          break;
LABEL_238:
        if ( v117 == v79 )
          goto LABEL_239;
      }
      while ( 2 )
      {
        while ( 1 )
        {
          v83 = v81;
          v81 = *(_QWORD *)(v81 + 8);
          v84 = *(unsigned __int8 *)(v83 - 24);
          if ( v84 != 50 )
            break;
          v85 = *(_BYTE **)(v83 - 88);
          if ( !v85 )
            BUG();
          if ( *v85 <= 0x15u )
            goto LABEL_219;
          v86 = *(_QWORD *)(v83 - 56);
          if ( !v86 )
            BUG();
          if ( *(_BYTE *)v86 != 18 )
            goto LABEL_219;
          v114 = *(__int64 **)(v86 + 24);
          v87 = (__int64 *)sub_C33340();
          v88 = (__int64)v114;
          if ( v114 == v87 )
          {
            v115 = v87;
            sub_C3C5A0(&v128, (__int64)v87, 1);
          }
          else
          {
            v115 = v87;
            sub_C36740((__int64)&v128, v88, 1);
          }
          v89 = v86 + 24;
          if ( v128 == v115 )
            v90 = sub_C3EF50(&v128, v89, 1u);
          else
            v90 = sub_C3B6C0((__int64)&v128, v89, 1);
          v91 = v115;
          if ( (v90 & 0xFFFFFFEF) == 0 )
          {
            v92 = (_QWORD *)(v83 - 24);
            v93 = *(_BYTE *)(*(_QWORD *)(v83 - 16) + 8LL);
            if ( v93 == 3 )
            {
              sub_2CD3AE0(&v124, v92, (__int64)"__nv_fdiv_by_const_dp", 0x15u, (__int64 *)&v128);
              v91 = v115;
            }
            else if ( v93 == 2 )
            {
              sub_2CD3AE0(&v124, v92, (__int64)"__nv_fdiv_by_const_sp", 0x15u, (__int64 *)&v128);
              v91 = v115;
            }
          }
          if ( v91 == v128 )
          {
            if ( v129 )
            {
              v94 = 24LL * *(_QWORD *)(v129 - 8);
              for ( i = (_QWORD *)(v129 + v94); (_QWORD *)v129 != i; sub_91D830(i) )
                i -= 3;
              j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
            }
            goto LABEL_219;
          }
          sub_C338F0((__int64)&v128);
          if ( v82 == v81 )
            goto LABEL_238;
        }
        if ( (unsigned int)(v84 - 29) <= 0x15 )
        {
          if ( (unsigned int)(v84 - 30) > 0x13 )
            goto LABEL_255;
        }
        else if ( (unsigned int)(v84 - 51) > 0x2D )
        {
LABEL_255:
          BUG();
        }
LABEL_219:
        if ( v82 == v81 )
          goto LABEL_238;
        continue;
      }
    }
  }
LABEL_10:
  v12 = a1 + 4;
  v13 = a1 + 10;
  if ( v109 )
  {
    v124 = (__int64 *)&unk_443E780;
    v128 = (__int64 *)&v124;
    v125 = 49900;
    sub_CF15A0(a3, &v128, 1, a4);
    v128 = (__int64 *)&unk_43A2FC0;
    v129 = 636836;
    v121[0] = (__int64 *)&v128;
    sub_CF15A0(a3, v121, 1, a4);
    memset(a1, 0, 0x60u);
    a1[1] = v12;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v13;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
LABEL_11:
    a1[2] = 0x100000002LL;
    a1[1] = v12;
    a1[6] = 0;
    a1[7] = v13;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    a1[4] = &qword_4F82400;
  }
  return a1;
}
