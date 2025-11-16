// Function: sub_33A4350
// Address: 0x33a4350
//
void __fastcall sub_33A4350(__int64 a1, __int64 a2)
{
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rsi
  const __m128i *v7; // rax
  __int64 v8; // r15
  __m128i v9; // xmm0
  unsigned int v10; // eax
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r9
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r13
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int8 v27; // r13
  char v28; // al
  __int32 v29; // r10d
  char v30; // al
  __int32 v31; // edx
  __int32 v32; // edx
  __int32 v33; // edx
  __int32 v34; // edx
  __int32 v35; // edx
  __int64 v36; // rax
  __int64 v37; // rsi
  __int32 v38; // r11d
  unsigned int v39; // ecx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 *v44; // rax
  _QWORD *v45; // rax
  _OWORD *v46; // rdi
  __int64 (__fastcall *v47)(__int64, __int64, unsigned int); // rcx
  __int64 *v48; // rdi
  __int64 v49; // rax
  _DWORD *v50; // rax
  __int32 v51; // r11d
  __int32 v52; // r10d
  int v53; // edx
  unsigned __int16 v54; // ax
  __int64 v55; // r8
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r9
  unsigned __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 (*v62)(); // rax
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // r15
  __int32 v68; // r10d
  __int64 v69; // r8
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 *v73; // rax
  int v74; // edx
  __m128i v75; // rax
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  __int32 v80; // r11d
  unsigned int v81; // eax
  unsigned int v82; // eax
  __int32 v83; // r11d
  __int32 v84; // r10d
  __int64 v85; // rsi
  __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // r8
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r9
  unsigned __int64 v92; // rdx
  __int64 *v93; // rax
  __int64 v94; // rax
  __int64 *v95; // rax
  unsigned int v96; // eax
  char v97; // al
  __int64 v98; // [rsp-10h] [rbp-100h]
  unsigned int v99; // [rsp+0h] [rbp-F0h]
  __int32 v100; // [rsp+0h] [rbp-F0h]
  __int64 v101; // [rsp+0h] [rbp-F0h]
  __int64 v102; // [rsp+8h] [rbp-E8h]
  __int64 v103; // [rsp+10h] [rbp-E0h]
  __int64 (__fastcall *v104)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-E0h]
  __int32 v105; // [rsp+10h] [rbp-E0h]
  __int32 v106; // [rsp+10h] [rbp-E0h]
  __int32 v107; // [rsp+10h] [rbp-E0h]
  __int64 v108; // [rsp+10h] [rbp-E0h]
  __int64 v109; // [rsp+18h] [rbp-D8h]
  __m128i v110; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v111; // [rsp+30h] [rbp-C0h]
  __int64 v112; // [rsp+38h] [rbp-B8h]
  _OWORD **v113; // [rsp+48h] [rbp-A8h]
  __int64 v114; // [rsp+58h] [rbp-98h] BYREF
  __int64 v115; // [rsp+60h] [rbp-90h] BYREF
  int v116; // [rsp+68h] [rbp-88h]
  _OWORD *v117; // [rsp+70h] [rbp-80h] BYREF
  __int64 v118; // [rsp+78h] [rbp-78h]
  _OWORD v119[7]; // [rsp+80h] [rbp-70h] BYREF

  v4 = *(_DWORD *)(a1 + 848);
  v5 = *(_QWORD *)a1;
  v115 = 0;
  v116 = v4;
  if ( v5 )
  {
    if ( &v115 != (__int64 *)(v5 + 48) )
    {
      v6 = *(_QWORD *)(v5 + 48);
      v115 = v6;
      if ( v6 )
        sub_B96E90((__int64)&v115, v6, 1);
    }
  }
  v7 = *(const __m128i **)(a1 + 864);
  v8 = 0;
  v117 = v119;
  v9 = _mm_loadu_si128(v7 + 24);
  v118 = 0x400000001LL;
  v119[0] = v9;
  v10 = sub_B5A050((unsigned __int8 *)a2);
  v113 = &v117;
  v11 = v10;
  if ( v10 )
  {
    do
    {
      v12 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (v8 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
      v13 = (unsigned int)v118;
      v15 = v14;
      v16 = (unsigned int)v118 + 1LL;
      if ( v16 > HIDWORD(v118) )
      {
        v111 = v12;
        v112 = v15;
        sub_C8D5F0((__int64)v113, v119, v16, 0x10u, v12, v15);
        v13 = (unsigned int)v118;
        v12 = v111;
        v15 = v112;
      }
      ++v8;
      v17 = (__int64 *)&v117[v13];
      *v17 = v12;
      v17[1] = v15;
      LODWORD(v118) = v118 + 1;
    }
    while ( v8 != v11 );
  }
  v18 = *(_QWORD *)(a1 + 864);
  v19 = *(__int64 **)(a2 + 8);
  v20 = *(_QWORD *)(v18 + 16);
  v21 = sub_2E79000(*(__int64 **)(v18 + 40));
  v22 = sub_2D5BAE0(v20, v21, v19, 0);
  v23 = v22;
  v99 = v22;
  v103 = v24;
  v25 = sub_33E5110(*(_QWORD *)(a1 + 864), v22, v24, 1, 0);
  v111 = v26;
  v113 = (_OWORD **)v25;
  v27 = sub_B59EF0((unsigned __int8 *)a2, v23);
  v110.m128i_i32[0] = (v27 == 0) << 12;
  v28 = sub_920620(a2);
  v29 = v110.m128i_i32[0];
  if ( v28 )
  {
    v30 = *(_BYTE *)(a2 + 1) >> 1;
    if ( (v30 & 2) != 0 )
      v29 = v110.m128i_i32[0] | 0x20;
    if ( (v30 & 4) != 0 )
      v29 |= 0x40u;
    v31 = v29;
    if ( (v30 & 8) != 0 )
    {
      LOBYTE(v31) = v29 | 0x80;
      v29 = v31;
    }
    v32 = v29;
    if ( (v30 & 0x10) != 0 )
    {
      BYTE1(v32) = BYTE1(v29) | 1;
      v29 = v32;
    }
    v33 = v29;
    if ( (v30 & 0x20) != 0 )
    {
      BYTE1(v33) = BYTE1(v29) | 2;
      v29 = v33;
    }
    v34 = v29;
    if ( (v30 & 0x40) != 0 )
    {
      BYTE1(v34) = BYTE1(v29) | 4;
      v29 = v34;
    }
    v35 = v29;
    if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
    {
      BYTE1(v35) = BYTE1(v29) | 8;
      v29 = v35;
    }
  }
  v36 = *(_QWORD *)(a2 - 32);
  if ( !v36 || *(_BYTE *)v36 || (v37 = *(_QWORD *)(a2 + 80), *(_QWORD *)(v36 + 24) != v37) )
    BUG();
  v38 = *(_DWORD *)(v36 + 36);
  switch ( v38 )
  {
    case 93:
      v39 = v118;
      v38 = 115;
      goto LABEL_29;
    case 94:
      v39 = v118;
      v38 = 114;
      goto LABEL_29;
    case 95:
      v39 = v118;
      v38 = 116;
      goto LABEL_29;
    case 96:
      v39 = v118;
      v38 = 117;
      goto LABEL_29;
    case 97:
      v39 = v118;
      v38 = 130;
      goto LABEL_29;
    case 98:
      v39 = v118;
      v38 = 112;
      goto LABEL_29;
    case 99:
      v39 = v118;
      v38 = 119;
      goto LABEL_29;
    case 100:
      v39 = v118;
      v38 = 121;
      goto LABEL_29;
    case 101:
      v39 = v118;
      v38 = 122;
      goto LABEL_29;
    case 102:
      v39 = v118;
      v38 = 101;
      goto LABEL_29;
    case 103:
      v80 = 147;
      goto LABEL_104;
    case 104:
      v80 = 148;
LABEL_104:
      v106 = v29;
      v110.m128i_i32[0] = v80;
      sub_B5A030(a2, v37);
      v82 = sub_34B9180(v81);
      v83 = v110.m128i_i32[0];
      v84 = v106;
      v85 = v82;
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 856) + 864LL) & 4) != 0 )
      {
        v96 = sub_34B9190(v82);
        v84 = v106;
        v83 = v110.m128i_i32[0];
        v85 = v96;
      }
      v86 = *(_QWORD *)(a1 + 864);
      v107 = v84;
      v110.m128i_i32[0] = v83;
      v87 = sub_33ED040(v86, v85);
      v38 = v110.m128i_i32[0];
      v88 = v87;
      v89 = (unsigned int)v118;
      v91 = v90;
      v29 = v107;
      v92 = (unsigned int)v118 + 1LL;
      if ( v92 > HIDWORD(v118) )
      {
        v101 = v88;
        v102 = v91;
        sub_C8D5F0((__int64)&v117, v119, v92, 0x10u, v88, v91);
        v89 = (unsigned int)v118;
        v88 = v101;
        v91 = v102;
        v29 = v107;
        v38 = v110.m128i_i32[0];
      }
      v93 = (__int64 *)&v117[v89];
      *v93 = v88;
      v93[1] = v91;
      v39 = v118 + 1;
      LODWORD(v118) = v118 + 1;
      goto LABEL_29;
    case 105:
      v39 = v118;
      v38 = 104;
      goto LABEL_29;
    case 106:
      v39 = v118;
      v38 = 131;
      goto LABEL_29;
    case 107:
      goto LABEL_38;
    case 108:
      v39 = v118;
      v38 = 103;
      goto LABEL_29;
    case 109:
      v61 = *(_QWORD *)(a1 + 864);
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 856) + 952LL) == 2 )
        goto LABEL_94;
      v62 = *(__int64 (**)())(*(_QWORD *)v20 + 1608LL);
      if ( v62 == sub_2FE3540 )
        goto LABEL_94;
      v110.m128i_i32[0] = v29;
      v97 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v62)(v20, *(_QWORD *)(v61 + 40), v99, v103);
      v29 = v110.m128i_i32[0];
      if ( v97 )
      {
LABEL_38:
        v39 = v118;
        v38 = 106;
      }
      else
      {
        v61 = *(_QWORD *)(a1 + 864);
LABEL_94:
        v110.m128i_i32[0] = v29;
        LODWORD(v118) = v118 - 1;
        v63 = sub_3410740(v61, 103, (unsigned int)&v115, (_DWORD)v113, v111, v29, (__int64)v117, (unsigned int)v118);
        v66 = v98;
        v67 = v63;
        v68 = v110.m128i_i32[0];
        if ( v27 > 1u )
        {
          if ( v27 == 2 )
          {
            sub_3050D50(a1 + 704, v63, 1, v64, v65, v98);
            v68 = v110.m128i_i32[0];
          }
        }
        else
        {
          sub_3050D50(a1 + 560, v63, 1, v64, v65, v98);
          v68 = v110.m128i_i32[0];
        }
        v69 = HIDWORD(v118);
        LODWORD(v118) = 0;
        v70 = 0;
        if ( !HIDWORD(v118) )
        {
          v110.m128i_i32[0] = v68;
          sub_C8D5F0((__int64)&v117, v119, 1u, 0x10u, 0, v66);
          v68 = v110.m128i_i32[0];
          v70 = (unsigned int)v118;
        }
        v71 = (__int64)v117;
        *(_QWORD *)&v117[v70] = v67;
        *(_QWORD *)(v71 + v70 * 16 + 8) = 1;
        LODWORD(v118) = v118 + 1;
        v72 = (unsigned int)v118;
        if ( (unsigned __int64)(unsigned int)v118 + 1 > HIDWORD(v118) )
        {
          v110.m128i_i32[0] = v68;
          sub_C8D5F0((__int64)&v117, v119, (unsigned int)v118 + 1LL, 0x10u, v69, v66);
          v72 = (unsigned int)v118;
          v68 = v110.m128i_i32[0];
        }
        v73 = (__int64 *)&v117[v72];
        v105 = v68;
        *v73 = v67;
        v73[1] = 0;
        v74 = *(_DWORD *)(a2 + 4);
        LODWORD(v118) = v118 + 1;
        v75.m128i_i64[0] = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (2LL - (v74 & 0x7FFFFFF))));
        v29 = v105;
        v110 = v75;
        v78 = (unsigned int)v118;
        v79 = (unsigned int)v118 + 1LL;
        if ( v79 > HIDWORD(v118) )
        {
          sub_C8D5F0((__int64)&v117, v119, v79, 0x10u, v76, v77);
          v78 = (unsigned int)v118;
          v29 = v105;
        }
        v38 = 101;
        v117[v78] = _mm_load_si128(&v110);
        v39 = v118 + 1;
        LODWORD(v118) = v118 + 1;
      }
LABEL_29:
      v42 = sub_3410740(*(_QWORD *)(a1 + 864), v38, (unsigned int)&v115, (_DWORD)v113, v111, v29, (__int64)v117, v39);
      if ( v27 > 1u )
      {
        if ( v27 == 2 )
        {
          v94 = *(unsigned int *)(a1 + 712);
          if ( v94 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 716) )
          {
            sub_C8D5F0(a1 + 704, (const void *)(a1 + 720), v94 + 1, 0x10u, v40, v41);
            v94 = *(unsigned int *)(a1 + 712);
          }
          v95 = (__int64 *)(*(_QWORD *)(a1 + 704) + 16 * v94);
          *v95 = v42;
          v95[1] = 1;
          ++*(_DWORD *)(a1 + 712);
        }
      }
      else
      {
        v43 = *(unsigned int *)(a1 + 568);
        if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 572) )
        {
          sub_C8D5F0(a1 + 560, (const void *)(a1 + 576), v43 + 1, 0x10u, v40, v41);
          v43 = *(unsigned int *)(a1 + 568);
        }
        v44 = (__int64 *)(*(_QWORD *)(a1 + 560) + 16 * v43);
        *v44 = v42;
        v44[1] = 1;
        ++*(_DWORD *)(a1 + 568);
      }
      v114 = a2;
      v45 = sub_337DC20(a1 + 8, &v114);
      *v45 = v42;
      v46 = v117;
      *((_DWORD *)v45 + 2) = 0;
      if ( v46 != v119 )
        _libc_free((unsigned __int64)v46);
      if ( v115 )
        sub_B91220((__int64)&v115, v115);
      return;
    case 110:
      v39 = v118;
      v38 = 146;
      goto LABEL_29;
    case 111:
      v39 = v118;
      v38 = 141;
      goto LABEL_29;
    case 112:
      v39 = v118;
      v38 = 142;
      goto LABEL_29;
    case 113:
      v100 = v29;
      v47 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v20 + 32LL);
      v48 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
      v110.m128i_i64[0] = *(_QWORD *)(a1 + 864);
      v104 = v47;
      v49 = sub_2E79000(v48);
      if ( v104 == sub_2D42F30 )
      {
        v50 = sub_AE2980(v49, 0);
        v51 = v110.m128i_i32[0];
        v52 = v100;
        v53 = v50[1];
        v54 = 2;
        if ( v53 != 1 )
        {
          v54 = 3;
          if ( v53 != 2 )
          {
            v54 = 4;
            if ( v53 != 4 )
            {
              v54 = 5;
              if ( v53 != 8 )
              {
                v54 = 6;
                if ( v53 != 16 )
                {
                  v54 = 7;
                  if ( v53 != 32 )
                  {
                    v54 = 8;
                    if ( v53 != 64 )
                      v54 = 9 * (v53 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v54 = v104(v20, v49, 0);
        v52 = v100;
        v51 = v110.m128i_i32[0];
      }
      v110.m128i_i32[0] = v52;
      v55 = sub_3400BD0(v51, 0, (unsigned int)&v115, v54, 0, 1, 0);
      v56 = (unsigned int)v118;
      v58 = v57;
      v29 = v110.m128i_i32[0];
      v59 = (unsigned int)v118 + 1LL;
      if ( v59 > HIDWORD(v118) )
      {
        v108 = v55;
        v109 = v58;
        sub_C8D5F0((__int64)&v117, v119, v59, 0x10u, v55, v58);
        v56 = (unsigned int)v118;
        v55 = v108;
        v58 = v109;
        v29 = v110.m128i_i32[0];
      }
      v60 = (__int64 *)&v117[v56];
      v38 = 145;
      *v60 = v55;
      v60[1] = v58;
      v39 = v118 + 1;
      LODWORD(v118) = v118 + 1;
      goto LABEL_29;
    case 114:
      v39 = v118;
      v38 = 105;
      goto LABEL_29;
    case 115:
      v39 = v118;
      v38 = 102;
      goto LABEL_29;
    case 116:
      v39 = v118;
      v38 = 110;
      goto LABEL_29;
    case 117:
      v39 = v118;
      v38 = 138;
      goto LABEL_29;
    case 118:
      v39 = v118;
      v38 = 136;
      goto LABEL_29;
    case 119:
      v39 = v118;
      v38 = 123;
      goto LABEL_29;
    case 120:
      v39 = v118;
      v38 = 124;
      goto LABEL_29;
    case 121:
      v39 = v118;
      v38 = 125;
      goto LABEL_29;
    case 122:
      v39 = v118;
      v38 = 137;
      goto LABEL_29;
    case 123:
      v39 = v118;
      v38 = 135;
      goto LABEL_29;
    case 124:
      v39 = v118;
      v38 = 139;
      goto LABEL_29;
    case 125:
      v39 = v118;
      v38 = 128;
      goto LABEL_29;
    case 126:
      v39 = v118;
      v38 = 140;
      goto LABEL_29;
    case 127:
      v39 = v118;
      v38 = 129;
      goto LABEL_29;
    case 128:
      v39 = v118;
      v38 = 127;
      goto LABEL_29;
    case 129:
      v39 = v118;
      v38 = 108;
      goto LABEL_29;
    case 130:
      v39 = v118;
      v38 = 109;
      goto LABEL_29;
    case 131:
      v39 = v118;
      v38 = 126;
      goto LABEL_29;
    case 132:
    case 133:
      v39 = v118;
      goto LABEL_29;
    case 134:
      v39 = v118;
      v38 = 111;
      goto LABEL_29;
    case 135:
      v39 = v118;
      v38 = 118;
      goto LABEL_29;
    case 136:
      v39 = v118;
      v38 = 143;
      goto LABEL_29;
    case 137:
      v39 = v118;
      v38 = 107;
      goto LABEL_29;
    case 138:
      v39 = v118;
      v38 = 113;
      goto LABEL_29;
    case 139:
      v39 = v118;
      v38 = 120;
      goto LABEL_29;
    case 140:
      v39 = v118;
      v38 = 134;
      goto LABEL_29;
    case 141:
      v39 = v118;
      v38 = 144;
      goto LABEL_29;
    default:
      BUG();
  }
}
