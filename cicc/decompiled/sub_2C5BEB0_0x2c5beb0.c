// Function: sub_2C5BEB0
// Address: 0x2c5beb0
//
__int64 __fastcall sub_2C5BEB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r15
  unsigned __int64 v9; // r13
  char v10; // dl
  __m128i v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // edx
  int v15; // r12d
  __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // edx
  bool v21; // zf
  int v22; // edx
  __int64 v23; // rax
  int v24; // edx
  int v25; // edx
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  int v28; // edx
  int v29; // edx
  bool v30; // of
  unsigned __int64 v31; // r13
  __int64 v32; // r15
  unsigned __int8 *v33; // r14
  unsigned int v34; // edx
  unsigned __int8 **v35; // rcx
  unsigned __int8 *v36; // rdi
  _QWORD *v37; // rdi
  __int64 v38; // rax
  const char *v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rax
  char v44; // al
  char v45; // di
  _QWORD *v46; // rax
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // r13
  __int64 v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  unsigned __int64 v55; // rax
  __int64 k; // r13
  __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // rbx
  int v60; // r14d
  __int64 v61; // rdi
  __int64 v62; // rcx
  __int64 *v63; // rdx
  __int64 v64; // rdi
  _QWORD *v65; // rcx
  _QWORD *v66; // rdx
  _QWORD *v67; // rax
  int v68; // r11d
  __int64 *v69; // rax
  int v70; // edx
  int v71; // ecx
  int v72; // r9d
  int v73; // edi
  __int64 *v74; // rcx
  unsigned int j; // edx
  __int64 v76; // r9
  int v77; // r8d
  unsigned int i; // r9d
  __int64 *v79; // rcx
  __int64 v80; // rdx
  bool v81; // cc
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rax
  unsigned int v84; // edx
  unsigned int v85; // r9d
  __int64 v86; // [rsp-10h] [rbp-130h]
  __int64 v87; // [rsp+10h] [rbp-110h]
  __int64 v88; // [rsp+20h] [rbp-100h]
  __int64 v89; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v90; // [rsp+30h] [rbp-F0h]
  __int64 v91; // [rsp+30h] [rbp-F0h]
  __int64 v92; // [rsp+38h] [rbp-E8h]
  int v93; // [rsp+40h] [rbp-E0h]
  __int64 v94; // [rsp+40h] [rbp-E0h]
  __int64 v95; // [rsp+40h] [rbp-E0h]
  signed __int64 v96; // [rsp+48h] [rbp-D8h]
  signed __int64 v97; // [rsp+50h] [rbp-D0h]
  __int64 v98; // [rsp+58h] [rbp-C8h]
  __int64 v99; // [rsp+58h] [rbp-C8h]
  char v100; // [rsp+60h] [rbp-C0h]
  __int64 v101; // [rsp+60h] [rbp-C0h]
  __int64 v102; // [rsp+60h] [rbp-C0h]
  int v103; // [rsp+68h] [rbp-B8h]
  int v104; // [rsp+68h] [rbp-B8h]
  __int64 v105; // [rsp+68h] [rbp-B8h]
  __int64 v106; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v107; // [rsp+78h] [rbp-A8h]
  __int64 v108; // [rsp+80h] [rbp-A0h]
  unsigned int v109; // [rsp+88h] [rbp-98h]
  _BYTE *v110; // [rsp+90h] [rbp-90h] BYREF
  __int64 v111; // [rsp+98h] [rbp-88h]
  char *v112; // [rsp+A0h] [rbp-80h]
  __int16 v113; // [rsp+B0h] [rbp-70h]
  __m128i v114; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v115; // [rsp+E0h] [rbp-40h]

  if ( *(_BYTE *)a2 == 61
    && ((v4 = a1, v5 = a2, (*(_BYTE *)(a2 + 7) & 0x40) != 0)
      ? (v6 = *(__int64 **)(a2 - 8))
      : (v6 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
        (v92 = *v6) != 0 && (v2 = *(_BYTE *)(a2 + 2) & 1, (*(_BYTE *)(a2 + 2) & 1) == 0)) )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(_QWORD *)(a1 + 184);
    v88 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
      v7 = **(_QWORD **)(v7 + 16);
    v9 = (sub_9208B0(v8, v7) + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v100 = v10;
    v11.m128i_i64[0] = sub_9208B0(v8, v7);
    v114 = v11;
    if ( v11.m128i_i64[0] == v9 && v114.m128i_i8[8] == v100 )
    {
      v12 = sub_DFD4A0(*(__int64 **)(a1 + 152));
      v106 = 0;
      v96 = v12;
      v13 = *(_QWORD *)(v5 + 16);
      v93 = v14;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      v98 = v13;
      if ( v13 )
      {
        v89 = v5;
        v97 = 0;
        v103 = 0;
        v101 = v5;
        v90 = v2;
        v15 = 0;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v98 + 24);
          if ( *(_BYTE *)v16 != 90 || *(_QWORD *)(v16 + 40) != *(_QWORD *)(v101 + 40) || !*(_QWORD *)(v16 + 16) )
            goto LABEL_15;
          if ( sub_B445A0(v89, *(_QWORD *)(v98 + 24)) )
          {
            v57 = v16 + 24;
            if ( v16 + 24 == *(_QWORD *)(v101 + 32) )
            {
              v89 = v16;
            }
            else
            {
              v89 = v16;
              v58 = *(_QWORD *)(v101 + 32);
              v87 = v4;
              v59 = v57;
              v60 = qword_5010AC8;
              do
              {
                v61 = v58 - 24;
                if ( !v58 )
                  v61 = 0;
                if ( v60 == v15 || (unsigned __int8)sub_B46490(v61) )
                  goto LABEL_15;
                v58 = *(_QWORD *)(v58 + 8);
                ++v15;
              }
              while ( v59 != v58 );
              v16 = v89;
              v4 = v87;
            }
          }
          sub_2C50240(
            (__int64)&v114,
            *(_DWORD *)(v88 + 32),
            *(_QWORD *)(v16 - 32),
            v101,
            *(_QWORD *)(v4 + 176),
            *(_QWORD *)(v4 + 160));
          if ( !v114.m128i_i32[0] )
          {
LABEL_15:
            v2 = v90;
            goto LABEL_16;
          }
          if ( v114.m128i_i32[0] != 2 )
            goto LABEL_24;
          if ( !v109 )
            break;
          v62 = (v109 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v63 = &v107[3 * v62];
          v64 = *v63;
          if ( v16 != *v63 )
          {
            v68 = 1;
            v69 = 0;
            while ( v64 != -4096 )
            {
              if ( v64 == -8192 && !v69 )
                v69 = v63;
              LODWORD(v62) = (v109 - 1) & (v68 + v62);
              v63 = &v107[3 * (unsigned int)v62];
              v64 = *v63;
              if ( v16 == *v63 )
                goto LABEL_70;
              ++v68;
            }
            if ( !v69 )
              v69 = v63;
            ++v106;
            v70 = v108 + 1;
            if ( 4 * ((int)v108 + 1) < 3 * v109 )
            {
              if ( v109 - HIDWORD(v108) - v70 <= v109 >> 3 )
              {
                sub_2C4D9A0((__int64)&v106, v109);
                if ( !v109 )
                {
LABEL_140:
                  LODWORD(v108) = v108 + 1;
                  BUG();
                }
                v77 = 1;
                v69 = 0;
                for ( i = (v109 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)); ; i = (v109 - 1) & v85 )
                {
                  v79 = &v107[3 * i];
                  v80 = *v79;
                  if ( v16 == *v79 )
                  {
                    v70 = v108 + 1;
                    v69 = &v107[3 * i];
                    goto LABEL_92;
                  }
                  if ( v80 == -4096 )
                    break;
                  if ( v80 != -8192 || v69 )
                    v79 = v69;
                  v85 = v77 + i;
                  v69 = v79;
                  ++v77;
                }
                if ( !v69 )
                  v69 = &v107[3 * i];
                v70 = v108 + 1;
              }
              goto LABEL_92;
            }
LABEL_102:
            sub_2C4D9A0((__int64)&v106, 2 * v109);
            if ( !v109 )
              goto LABEL_140;
            v73 = 1;
            v74 = 0;
            for ( j = (v109 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)); ; j = (v109 - 1) & v84 )
            {
              v69 = &v107[3 * j];
              v76 = *v69;
              if ( v16 == *v69 )
              {
                v70 = v108 + 1;
                goto LABEL_92;
              }
              if ( v76 == -4096 )
                break;
              if ( v74 || v76 != -8192 )
                v69 = v74;
              v84 = v73 + j;
              v74 = v69;
              ++v73;
            }
            if ( v74 )
              v69 = v74;
            v70 = v108 + 1;
LABEL_92:
            LODWORD(v108) = v70;
            if ( *v69 != -4096 )
              --HIDWORD(v108);
            *v69 = v16;
            *(__m128i *)(v69 + 1) = _mm_loadu_si128(&v114);
          }
LABEL_70:
          v114.m128i_i64[1] = 0;
          v114.m128i_i32[0] = 0;
LABEL_24:
          v19 = sub_DFD330(*(__int64 **)(v4 + 152));
          v21 = v20 == 1;
          v22 = 1;
          if ( !v21 )
            v22 = v93;
          v93 = v22;
          if ( __OFADD__(v19, v96) )
          {
            v81 = v19 <= 0;
            v83 = 0x8000000000000000LL;
            if ( !v81 )
              v83 = 0x7FFFFFFFFFFFFFFFLL;
            v96 = v83;
          }
          else
          {
            v96 += v19;
          }
          v23 = sub_DFD4A0(*(__int64 **)(v4 + 152));
          v21 = v24 == 1;
          v25 = 1;
          if ( !v21 )
            v25 = v103;
          v26 = v23 + v97;
          v104 = v25;
          if ( __OFADD__(v23, v97) )
          {
            v26 = 0x8000000000000000LL;
            if ( v23 > 0 )
              v26 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v27 = sub_DFDB90(*(_QWORD *)(v4 + 152));
          v21 = v28 == 1;
          v29 = 1;
          if ( !v21 )
            v29 = v104;
          v30 = __OFADD__(v27, v26);
          v31 = v27 + v26;
          v103 = v29;
          if ( v30 )
          {
            v81 = v27 <= 0;
            v82 = 0x8000000000000000LL;
            if ( !v81 )
              v82 = 0x7FFFFFFFFFFFFFFFLL;
            v97 = v82;
          }
          else
          {
            v97 = v31;
          }
          v98 = *(_QWORD *)(v98 + 8);
          if ( !v98 )
          {
            v2 = v90;
            v5 = v101;
            goto LABEL_37;
          }
        }
        ++v106;
        goto LABEL_102;
      }
      v97 = 0;
      v103 = 0;
LABEL_37:
      if ( v93 == v103 )
      {
        if ( v96 > v97 )
          goto LABEL_39;
      }
      else if ( v103 < v93 )
      {
LABEL_39:
        v32 = v4 + 200;
        sub_F15FC0(v4 + 200, v5);
        v102 = *(_QWORD *)(v5 + 16);
        if ( v102 )
        {
          v91 = v5;
          v99 = v4 + 8;
          do
          {
            v33 = *(unsigned __int8 **)(v102 + 24);
            v105 = *((_QWORD *)v33 - 4);
            if ( v109 )
            {
              v34 = (v109 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v35 = (unsigned __int8 **)&v107[3 * v34];
              v36 = *v35;
              if ( v33 == *v35 )
              {
LABEL_43:
                if ( v35 != &v107[3 * v109] )
                  sub_2C52B50((__int64)(v35 + 1), v99, v105);
              }
              else
              {
                v71 = 1;
                while ( v36 != (unsigned __int8 *)-4096LL )
                {
                  v72 = v71 + 1;
                  v34 = (v109 - 1) & (v71 + v34);
                  v35 = (unsigned __int8 **)&v107[3 * v34];
                  v36 = *v35;
                  if ( v33 == *v35 )
                    goto LABEL_43;
                  v71 = v72;
                }
              }
            }
            sub_D5F1F0(v99, (__int64)v33);
            v37 = *(_QWORD **)(v4 + 80);
            v115 = 257;
            v38 = sub_BCB2D0(v37);
            v110 = (_BYTE *)sub_ACD640(v38, 0, 0);
            v111 = v105;
            v94 = sub_921130((unsigned int **)v99, v88, v92, &v110, 2, (__int64)&v114, 3u);
            v39 = sub_BD5D20((__int64)v33);
            v113 = 773;
            v40 = *(_QWORD *)(v4 + 56);
            v110 = v39;
            v111 = v41;
            v112 = ".scalar";
            v42 = *(_QWORD *)(v88 + 24);
            v43 = sub_AA4E30(v40);
            v44 = sub_AE5020(v43, v42);
            v115 = 257;
            v45 = v44;
            v46 = sub_BD2C40(80, 1u);
            v47 = v86;
            v48 = (__int64)v46;
            if ( v46 )
              sub_B4D190((__int64)v46, v42, v94, (__int64)&v114, 0, v45, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD, __int64))(**(_QWORD **)(v4 + 96) + 16LL))(
              *(_QWORD *)(v4 + 96),
              v48,
              &v110,
              *(_QWORD *)(v99 + 56),
              *(_QWORD *)(v99 + 64),
              v47);
            v49 = *(_QWORD *)(v4 + 8);
            v50 = 16LL * *(unsigned int *)(v4 + 16);
            if ( v49 != v49 + v50 )
            {
              v95 = v4;
              v51 = v49 + v50;
              v52 = *(_QWORD *)(v4 + 8);
              do
              {
                v53 = *(_QWORD *)(v52 + 8);
                v54 = *(_DWORD *)v52;
                v52 += 16;
                sub_B99FD0(v48, v54, v53);
              }
              while ( v51 != v52 );
              v4 = v95;
            }
            _BitScanReverse64(&v55, 1LL << (*(_WORD *)(v91 + 2) >> 1));
            *(_WORD *)(v48 + 2) = (2
                                 * (unsigned __int8)sub_2C514E0(
                                                      63 - ((unsigned __int8)v55 ^ 0x3Fu),
                                                      *(_QWORD *)(v88 + 24),
                                                      v105,
                                                      *(_QWORD *)(v4 + 184)))
                                | *(_WORD *)(v48 + 2) & 0xFF81;
            sub_BD84D0((__int64)v33, v48);
            if ( *(_BYTE *)v48 > 0x1Cu )
            {
              sub_BD6B90((unsigned __int8 *)v48, v33);
              for ( k = *(_QWORD *)(v48 + 16); k; k = *(_QWORD *)(k + 8) )
                sub_F15FC0(v32, *(_QWORD *)(k + 24));
              if ( *(_BYTE *)v48 > 0x1Cu )
                sub_F15FC0(v32, v48);
            }
            if ( *v33 > 0x1Cu )
              sub_F15FC0(v32, (__int64)v33);
            v102 = *(_QWORD *)(v102 + 8);
          }
          while ( v102 );
        }
        v17 = (__int64)v107;
        v2 = 1;
        v18 = 3LL * v109;
LABEL_17:
        sub_C7D6A0(v17, v18 * 8, 8);
        return v2;
      }
LABEL_16:
      v17 = (__int64)v107;
      v18 = 3LL * v109;
      if ( (_DWORD)v108 )
      {
        v65 = &v107[v18];
        if ( &v107[v18] != v107 )
        {
          v66 = v107;
          while ( 1 )
          {
            v67 = v66;
            if ( *v66 != -4096 && *v66 != -8192 )
              break;
            v66 += 3;
            if ( v65 == v66 )
              goto LABEL_17;
          }
          if ( v65 != v66 )
          {
            do
            {
              v67[2] = 0;
              v67 += 3;
              *((_DWORD *)v67 - 4) = 0;
              if ( v67 == v65 )
                break;
              while ( *v67 == -8192 || *v67 == -4096 )
              {
                v67 += 3;
                if ( v65 == v67 )
                  goto LABEL_81;
              }
            }
            while ( v67 != v65 );
LABEL_81:
            v17 = (__int64)v107;
            v18 = 3LL * v109;
          }
        }
      }
      goto LABEL_17;
    }
  }
  else
  {
    return 0;
  }
  return v2;
}
