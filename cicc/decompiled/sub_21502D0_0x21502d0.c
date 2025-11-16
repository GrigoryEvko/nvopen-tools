// Function: sub_21502D0
// Address: 0x21502d0
//
__int64 __fastcall sub_21502D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (*v7)(void); // rax
  unsigned int v8; // eax
  __int64 *v9; // r13
  __int64 *v10; // rdx
  __int64 v11; // r12
  int v12; // edx
  __int64 v13; // rax
  void *v14; // rdx
  unsigned __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v22; // rbx
  unsigned int v23; // r12d
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  _WORD *v29; // rdx
  char *v30; // rax
  size_t v31; // rdx
  char **v32; // r12
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // rbx
  _QWORD *v36; // rcx
  _QWORD *v37; // rbx
  int v38; // eax
  _QWORD *v39; // r12
  __int64 v40; // rbx
  void *v41; // r13
  __int64 v42; // r14
  _QWORD *v43; // rbx
  __int64 v44; // rax
  bool v45; // zf
  char v46; // al
  char v47; // al
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rbx
  int v52; // eax
  unsigned int v53; // ebx
  unsigned int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rdi
  const __m128i *v57; // r12
  __int8 v58; // al
  char v59; // cl
  unsigned int v60; // ebx
  unsigned int v61; // r12d
  char v62; // cl
  bool v63; // al
  _QWORD *v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rdi
  _BYTE *v67; // rax
  unsigned int v68; // r12d
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  _WORD *v72; // rdx
  void *v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // r8
  __int64 v77; // rax
  unsigned __int64 v78; // r8
  unsigned int v79; // edx
  __int64 v80; // r9
  unsigned int v81; // edx
  const char *v82; // rsi
  __int64 v83; // rax
  __int64 v84; // [rsp+10h] [rbp-1F0h]
  unsigned int v85; // [rsp+18h] [rbp-1E8h]
  __int16 v86; // [rsp+1Dh] [rbp-1E3h]
  char v87; // [rsp+1Fh] [rbp-1E1h]
  _QWORD *v88; // [rsp+20h] [rbp-1E0h]
  __int64 v89; // [rsp+20h] [rbp-1E0h]
  __int64 *v90; // [rsp+28h] [rbp-1D8h]
  __int64 v91; // [rsp+30h] [rbp-1D0h]
  char v93; // [rsp+40h] [rbp-1C0h]
  __int64 v94; // [rsp+40h] [rbp-1C0h]
  __int64 v95; // [rsp+48h] [rbp-1B8h]
  unsigned int v96; // [rsp+48h] [rbp-1B8h]
  unsigned __int64 v97; // [rsp+48h] [rbp-1B8h]
  _QWORD *v98; // [rsp+50h] [rbp-1B0h]
  unsigned int v99; // [rsp+50h] [rbp-1B0h]
  size_t v100; // [rsp+50h] [rbp-1B0h]
  unsigned __int64 v101; // [rsp+50h] [rbp-1B0h]
  __int64 v102; // [rsp+50h] [rbp-1B0h]
  unsigned __int64 v103; // [rsp+60h] [rbp-1A0h]
  unsigned __int64 v104; // [rsp+60h] [rbp-1A0h]
  int v105; // [rsp+60h] [rbp-1A0h]
  unsigned int v106; // [rsp+68h] [rbp-198h]
  unsigned int v107; // [rsp+6Ch] [rbp-194h]
  __int64 v108; // [rsp+78h] [rbp-188h] BYREF
  _BYTE *v109; // [rsp+80h] [rbp-180h] BYREF
  __int64 v110; // [rsp+88h] [rbp-178h]
  _QWORD v111[2]; // [rsp+90h] [rbp-170h] BYREF
  void *s2[2]; // [rsp+A0h] [rbp-160h] BYREF
  _QWORD v113[2]; // [rsp+B0h] [rbp-150h] BYREF
  char *v114; // [rsp+C0h] [rbp-140h] BYREF
  size_t v115; // [rsp+C8h] [rbp-138h]
  __int64 v116; // [rsp+D0h] [rbp-130h] BYREF
  void *dest; // [rsp+D8h] [rbp-128h]
  int v118; // [rsp+E0h] [rbp-120h]
  _BYTE **v119; // [rsp+E8h] [rbp-118h]

  v4 = a1;
  v5 = sub_396DDB0();
  v6 = a1[105];
  v91 = v5;
  v108 = *(_QWORD *)(a2 + 112);
  v7 = *(__int64 (**)(void))(*(_QWORD *)v6 + 56LL);
  if ( (char *)v7 == (char *)sub_214ABA0 )
    v84 = v6 + 696;
  else
    v84 = v7();
  HIBYTE(v86) = sub_1C2F070(a2);
  v85 = *(_DWORD *)(v4[105] + 252LL);
  LOBYTE(v86) = v85 > 0x13;
  v8 = 8 * sub_15A9520(v91, 0);
  if ( v8 == 32 )
  {
    v87 = 5;
  }
  else if ( v8 > 0x20 )
  {
    v87 = 6;
    if ( v8 != 64 )
    {
      v45 = v8 == 128;
      v46 = 7;
      if ( !v45 )
        v46 = 0;
      v87 = v46;
    }
  }
  else
  {
    v87 = 3;
    if ( v8 != 8 )
      v87 = 4 * (v8 == 16);
  }
  if ( !*(_QWORD *)(a2 + 96) && !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
    return sub_1263B40(a3, "()\n");
  sub_1263B40(a3, "(\n");
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, (__int64)"(\n");
    v9 = *(__int64 **)(a2 + 88);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      sub_15E08E0(a2, (__int64)"(\n");
    v10 = *(__int64 **)(a2 + 88);
  }
  else
  {
    v9 = *(__int64 **)(a2 + 88);
    v10 = v9;
  }
  v90 = &v10[5 * *(_QWORD *)(a2 + 96)];
  if ( v90 == v9 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8 )
      goto LABEL_31;
    return sub_1263B40(a3, "\n)\n");
  }
  v93 = 1;
  v107 = 0;
  do
  {
    v11 = *v9;
    if ( *(_BYTE *)(*v9 + 8) == 13 && (*(_BYTE *)(v11 + 9) & 1) == 0 )
    {
      ++v107;
      goto LABEL_27;
    }
    if ( !v93 )
    {
      v29 = *(_WORD **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v29 <= 1u )
      {
        sub_16E7EE0(a3, ",\n", 2u);
      }
      else
      {
        *v29 = 2604;
        *(_QWORD *)(a3 + 24) += 2LL;
      }
    }
    if ( (unsigned __int8)sub_1C2F070(a2)
      && ((unsigned __int8)sub_1C2E890((__int64)v9) || (unsigned __int8)sub_1C2EBB0((__int64)v9)) )
    {
      v109 = v111;
      v110 = 0;
      LOBYTE(v111[0]) = 0;
      v118 = 1;
      dest = 0;
      v116 = 0;
      v115 = 0;
      v114 = (char *)&unk_49EFBE0;
      v119 = &v109;
      v30 = (char *)sub_1649960(a2);
      if ( v31 > v116 - (__int64)dest )
      {
        v32 = (char **)sub_16E7EE0((__int64)&v114, v30, v31);
      }
      else
      {
        v32 = &v114;
        if ( v31 )
        {
          v100 = v31;
          memcpy(dest, v30, v31);
          dest = (char *)dest + v100;
        }
      }
      v33 = sub_1263B40((__int64)v32, "_param_");
      v34 = v107;
      sub_16E7A90(v33, v107);
      if ( (void *)v115 != dest )
        sub_16E7BA0((__int64 *)&v114);
      v35 = v4[33];
      v36 = *(_QWORD **)(v35 + 48);
      if ( !v36 )
      {
        v74 = *(_QWORD *)(v35 + 120);
        v75 = *(_QWORD *)(v35 + 128);
        *(_QWORD *)(v35 + 200) += 280LL;
        if ( ((v74 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v74 + 280 <= v75 - v74 )
        {
          v36 = (_QWORD *)((v74 + 7) & 0xFFFFFFFFFFFFFFF8LL);
          *(_QWORD *)(v35 + 120) = v36 + 35;
        }
        else
        {
          v76 = 0x40000000000LL;
          v96 = *(_DWORD *)(v35 + 144);
          if ( v96 >> 7 < 0x1E )
            v76 = 4096LL << (v96 >> 7);
          v101 = v76;
          v77 = malloc(v76);
          v78 = v101;
          v79 = v96;
          v80 = v77;
          if ( !v77 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v79 = *(_DWORD *)(v35 + 144);
            v78 = v101;
            v80 = 0;
          }
          if ( *(_DWORD *)(v35 + 148) <= v79 )
          {
            v97 = v78;
            v102 = v80;
            sub_16CD150(v35 + 136, (const void *)(v35 + 152), 0, 8, v78, v80);
            v79 = *(_DWORD *)(v35 + 144);
            v78 = v97;
            v80 = v102;
          }
          *(_QWORD *)(*(_QWORD *)(v35 + 136) + 8LL * v79) = v80;
          *(_QWORD *)(v35 + 128) = v80 + v78;
          ++*(_DWORD *)(v35 + 144);
          v36 = (_QWORD *)((v80 + 7) & 0xFFFFFFFFFFFFFFF8LL);
          *(_QWORD *)(v35 + 120) = v36 + 35;
        }
        *v36 = &unk_4A016E8;
        v36[1] = v36 + 3;
        v36[2] = 0x800000000LL;
        *(_QWORD *)(v35 + 48) = v36;
      }
      v98 = v36;
      s2[0] = v113;
      sub_214ADD0((__int64 *)s2, v109, (__int64)&v109[v110]);
      v37 = s2[0];
      v38 = *((_DWORD *)v98 + 4);
      if ( v38 )
      {
        v94 = (__int64)v9;
        v39 = s2[0];
        v88 = v4;
        v40 = v98[1];
        v41 = s2[1];
        v42 = v40 + 32LL * (unsigned int)(v38 - 1) + 32;
        while ( *(void **)(v40 + 8) != v41 || v41 && memcmp(*(const void **)v40, v39, (size_t)v41) )
        {
          v40 += 32;
          if ( v40 == v42 )
          {
            v37 = v39;
            v9 = (__int64 *)v94;
            v34 = v107;
            v4 = v88;
            goto LABEL_69;
          }
        }
        v43 = v39;
        v9 = (__int64 *)v94;
        v34 = v107;
        v4 = v88;
        if ( v43 != v113 )
          j_j___libc_free_0(v43, v113[0] + 1LL);
        if ( (unsigned __int8)sub_1C2EBB0(v94) )
        {
          if ( (unsigned __int8)sub_1C2EAF0(v94) || (unsigned __int8)sub_1C2EA30(v94) )
            sub_1263B40(a3, "\t.param .surfref ");
          else
            sub_1263B40(a3, "\t.param .texref ");
        }
        else
        {
          sub_1263B40(a3, "\t.param .samplerref ");
        }
      }
      else
      {
LABEL_69:
        if ( v37 != v113 )
          j_j___libc_free_0(v37, v113[0] + 1LL);
        if ( (unsigned __int8)sub_1C2EBB0((__int64)v9) )
        {
          if ( (unsigned __int8)sub_1C2EAF0((__int64)v9) || (unsigned __int8)sub_1C2EA30((__int64)v9) )
            sub_1263B40(a3, "\t.param .u64 .ptr .surfref ");
          else
            sub_1263B40(a3, "\t.param .u64 .ptr .texref ");
        }
        else
        {
          sub_1263B40(a3, "\t.param .u64 .ptr .samplerref ");
        }
      }
      sub_38E2490(v4[38], a3, v4[30]);
      v44 = sub_1263B40(a3, "_param_");
      sub_16E7A90(v44, v34);
      sub_16E7BC0((__int64 *)&v114);
      if ( v109 != (_BYTE *)v111 )
        j_j___libc_free_0(v109, v111[0] + 1LL);
      ++v107;
      v93 = 0;
    }
    else
    {
      v93 = sub_1560290(&v108, v107, 6);
      if ( v93 )
      {
        if ( *(_BYTE *)(v11 + 8) != 15 )
          BUG();
        v93 = v86 | HIBYTE(v86);
        v22 = *(_QWORD *)(v11 + 24);
        if ( v86 )
        {
          v23 = sub_15603A0(&v108, v107);
          if ( !v23 )
            v23 = sub_15A9FE0(v91, v22);
          if ( HIBYTE(v86) != 1 && v23 <= 3 )
            v23 = 4;
          v104 = (unsigned int)sub_15A9FE0(v91, v22);
          v24 = v104 * ((v104 + ((unsigned __int64)(sub_127FA20(v91, v22) + 7) >> 3) - 1) / v104);
          v25 = sub_1263B40(a3, "\t.param .align ");
          v26 = sub_16E7A90(v25, v23);
          sub_1263B40(v26, " .b8 ");
          sub_2150230((__int64)v4, (__int64)v9, v107, a3);
          v27 = sub_1263B40(a3, "[");
          v28 = sub_16E7A90(v27, (unsigned int)v24);
          sub_1263B40(v28, "]");
          ++v107;
          v93 = 0;
        }
        else
        {
          v114 = (char *)&v116;
          v115 = 0x1000000000LL;
          sub_20C7CE0(v84, v91, v22, (__int64)&v114, 0, 0);
          if ( (_DWORD)v115 )
          {
            v89 = (unsigned int)v115;
            v106 = v115 - 1;
            v95 = 0;
            while ( 1 )
            {
              v57 = (const __m128i *)&v114[16 * v95];
              *(__m128i *)s2 = _mm_loadu_si128(v57);
              v58 = v57->m128i_i8[0];
              if ( v57->m128i_i8[0] )
              {
                if ( (unsigned __int8)(v58 - 14) > 0x5Fu )
                  goto LABEL_89;
                v105 = word_4327020[(unsigned __int8)(v58 - 14)];
                switch ( v58 )
                {
                  case 24:
                  case 25:
                  case 26:
                  case 27:
                  case 28:
                  case 29:
                  case 30:
                  case 31:
                  case 32:
                  case 62:
                  case 63:
                  case 64:
                  case 65:
                  case 66:
                  case 67:
                    v59 = 3;
                    v73 = 0;
                    break;
                  case 33:
                  case 34:
                  case 35:
                  case 36:
                  case 37:
                  case 38:
                  case 39:
                  case 40:
                  case 68:
                  case 69:
                  case 70:
                  case 71:
                  case 72:
                  case 73:
                    v59 = 4;
                    v73 = 0;
                    break;
                  case 41:
                  case 42:
                  case 43:
                  case 44:
                  case 45:
                  case 46:
                  case 47:
                  case 48:
                  case 74:
                  case 75:
                  case 76:
                  case 77:
                  case 78:
                  case 79:
                    v59 = 5;
                    v73 = 0;
                    break;
                  case 49:
                  case 50:
                  case 51:
                  case 52:
                  case 53:
                  case 54:
                  case 80:
                  case 81:
                  case 82:
                  case 83:
                  case 84:
                  case 85:
                    v59 = 6;
                    v73 = 0;
                    break;
                  case 55:
                    v59 = 7;
                    v73 = 0;
                    break;
                  case 86:
                  case 87:
                  case 88:
                  case 98:
                  case 99:
                  case 100:
                    v59 = 8;
                    v73 = 0;
                    break;
                  case 89:
                  case 90:
                  case 91:
                  case 92:
                  case 93:
                  case 101:
                  case 102:
                  case 103:
                  case 104:
                  case 105:
                    v59 = 9;
                    v73 = 0;
                    break;
                  case 94:
                  case 95:
                  case 96:
                  case 97:
                  case 106:
                  case 107:
                  case 108:
                  case 109:
                    v59 = 10;
                    v73 = 0;
                    break;
                  default:
                    v59 = 2;
                    v73 = 0;
                    break;
                }
              }
              else
              {
                if ( !sub_1F58D20((__int64)&v114[16 * v95]) )
                {
LABEL_89:
                  v99 = 0;
                  v59 = (char)s2[0];
                  v105 = 1;
                  goto LABEL_90;
                }
                v105 = sub_1F58D30((__int64)v57);
                v59 = sub_1F596B0((__int64)v57);
              }
              LOBYTE(s2[0]) = v59;
              s2[1] = v73;
              if ( v105 )
                break;
LABEL_109:
              if ( v106 > (unsigned int)v95 )
                sub_1263B40(a3, ",\n");
              if ( v89 == ++v95 )
                goto LABEL_112;
            }
            v99 = v105 - 1;
LABEL_90:
            v60 = 0;
            while ( 2 )
            {
              if ( v59 )
              {
                v61 = sub_214AF00(v59);
                v63 = (unsigned __int8)(v62 - 14) <= 0x47u || (unsigned __int8)(v62 - 2) <= 5u;
              }
              else
              {
                v61 = sub_1F58D40((__int64)s2);
                v63 = sub_1F58CF0((__int64)s2);
              }
              if ( v63 && v61 < 0x20 )
                v61 = 32;
              v64 = *(_QWORD **)(a3 + 24);
              if ( *(_QWORD *)(a3 + 16) - (_QWORD)v64 <= 7u )
              {
                v65 = sub_16E7EE0(a3, "\t.reg .b", 8u);
              }
              else
              {
                v65 = a3;
                *v64 = 0x622E206765722E09LL;
                *(_QWORD *)(a3 + 24) += 8LL;
              }
              v66 = sub_16E7A90(v65, v61);
              v67 = *(_BYTE **)(v66 + 24);
              if ( *(_BYTE **)(v66 + 16) == v67 )
              {
                sub_16E7EE0(v66, " ", 1u);
              }
              else
              {
                *v67 = 32;
                ++*(_QWORD *)(v66 + 24);
              }
              v68 = v60 + v107;
              v69 = sub_396EAF0(v4, v9[3]);
              sub_38E2490(v69, a3, v4[30]);
              v70 = *(_QWORD *)(a3 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v70) <= 6 )
              {
                v71 = sub_16E7EE0(a3, "_param_", 7u);
                sub_16E7AB0(v71, (int)v68);
                if ( v99 <= v60 )
                  goto LABEL_101;
              }
              else
              {
                *(_BYTE *)(v70 + 6) = 95;
                *(_DWORD *)v70 = 1918988383;
                *(_WORD *)(v70 + 4) = 28001;
                *(_QWORD *)(a3 + 24) += 7LL;
                sub_16E7AB0(a3, (int)v68);
                if ( v99 <= v60 )
                {
LABEL_101:
                  if ( ++v60 == v105 )
                    goto LABEL_108;
                  goto LABEL_102;
                }
              }
              v72 = *(_WORD **)(a3 + 24);
              if ( *(_QWORD *)(a3 + 16) - (_QWORD)v72 <= 1u )
              {
                sub_16E7EE0(a3, ",\n", 2u);
                goto LABEL_101;
              }
              ++v60;
              *v72 = 2604;
              *(_QWORD *)(a3 + 24) += 2LL;
              if ( v60 == v105 )
              {
LABEL_108:
                v107 += v60;
                goto LABEL_109;
              }
LABEL_102:
              v59 = (char)s2[0];
              continue;
            }
          }
LABEL_112:
          if ( v114 != (char *)&v116 )
            _libc_free((unsigned __int64)v114);
        }
      }
      else
      {
        v12 = *(unsigned __int8 *)(v11 + 8);
        if ( (unsigned int)(v12 - 13) > 1 && (_BYTE)v12 != 16 && !sub_1642F90(v11, 128) )
        {
          v47 = *(_BYTE *)(v11 + 8);
          if ( v47 == 15 )
          {
            v48 = (unsigned int)sub_214AF00(v87);
            if ( HIBYTE(v86) )
            {
              v49 = sub_1263B40(a3, "\t.param .u");
              v50 = sub_16E7A90(v49, v48);
              sub_1263B40(v50, " ");
              if ( *(_DWORD *)(v4[29] + 952LL) == 1 )
              {
                v53 = sub_15E0370((__int64)v9);
                if ( v53 )
                {
                  sub_1263B40(a3, ".ptr .global ");
                  goto LABEL_84;
                }
              }
              else
              {
                v51 = *(_QWORD *)(v11 + 24);
                v52 = *(_DWORD *)(v11 + 8) >> 8;
                switch ( v52 )
                {
                  case 3:
                    sub_1263B40(a3, ".ptr .shared ");
                    break;
                  case 4:
                    sub_1263B40(a3, ".ptr .const ");
                    break;
                  case 1:
                    sub_1263B40(a3, ".ptr .global ");
                    break;
                  default:
                    sub_1263B40(a3, ".ptr ");
                    break;
                }
                v53 = sub_214B1D0(v91, v51);
                v54 = sub_15603A0(&v108, v107);
                if ( v53 < v54 )
                  v53 = v54;
LABEL_84:
                v55 = sub_1263B40(a3, ".align ");
                v56 = sub_16E7A90(v55, v53);
LABEL_161:
                sub_1263B40(v56, " ");
              }
              sub_2150230((__int64)v4, (__int64)v9, v107, a3);
              v93 = 0;
              ++v107;
              goto LABEL_27;
            }
          }
          else
          {
            if ( HIBYTE(v86) )
            {
              sub_1263B40(a3, "\t.param .");
              if ( sub_1642F90(v11, 1) )
              {
                sub_1263B40(a3, "u8");
              }
              else
              {
                sub_214FBF0((__int64)&v114, (__int64)v4, v11, 1);
                sub_16E7EE0(a3, v114, v115);
                sub_2240A30(&v114);
              }
              v56 = a3;
              goto LABEL_161;
            }
            if ( v47 == 11 )
            {
              v81 = *(_DWORD *)(v11 + 8) >> 8;
              if ( *(_DWORD *)(v11 + 8) <= 0x1FFFu )
                v81 = 32;
              v48 = v81;
            }
            else
            {
              v48 = 32;
              if ( v47 != 1 )
                v48 = (unsigned int)sub_1643030(v11);
            }
          }
          v82 = "\t.param .b";
          if ( v85 <= 0x13 )
            v82 = "\t.reg .b";
          v83 = sub_1263B40(a3, v82);
          v56 = sub_16E7A90(v83, v48);
          goto LABEL_161;
        }
        LODWORD(v114) = 0;
        if ( !(unsigned __int8)sub_1C2FF50(a2, v107 + 1, &v114) )
        {
          LODWORD(v114) = sub_15603A0(&v108, v107);
          if ( !(_DWORD)v114 )
            LODWORD(v114) = sub_15A9FE0(v91, v11);
        }
        v103 = (unsigned int)sub_15A9FE0(v91, v11);
        v13 = sub_127FA20(v91, v11);
        v14 = *(void **)(a3 + 24);
        v15 = v103 * ((v103 + ((unsigned __int64)(v13 + 7) >> 3) - 1) / v103);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v14 <= 0xEu )
        {
          v16 = sub_16E7EE0(a3, "\t.param .align ", 0xFu);
        }
        else
        {
          v16 = a3;
          qmemcpy(v14, "\t.param .align ", 15);
          *(_QWORD *)(a3 + 24) += 15LL;
        }
        v17 = sub_16E7A90(v16, (unsigned int)v114);
        sub_1263B40(v17, " .b8 ");
        sub_2150230((__int64)v4, (__int64)v9, v107, a3);
        v18 = *(_BYTE **)(a3 + 24);
        if ( *(_BYTE **)(a3 + 16) == v18 )
        {
          v19 = sub_16E7EE0(a3, "[", 1u);
        }
        else
        {
          *v18 = 91;
          v19 = a3;
          ++*(_QWORD *)(a3 + 24);
        }
        v20 = sub_16E7A90(v19, (unsigned int)v15);
        sub_1263B40(v20, "]");
        ++v107;
      }
    }
LABEL_27:
    v9 += 5;
  }
  while ( v9 != v90 );
  if ( *(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8 )
  {
    if ( !v93 )
      sub_1263B40(a3, ",\n");
LABEL_31:
    sub_1263B40(a3, "\t.param .align 8 .b8 %VAParam[]");
  }
  return sub_1263B40(a3, "\n)\n");
}
