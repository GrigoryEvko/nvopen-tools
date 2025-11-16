// Function: sub_1531F90
// Address: 0x1531f90
//
void __fastcall sub_1531F90(__int64 ***a1, unsigned int a2, unsigned int a3, char a4)
{
  __m128i *v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  signed __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 i; // rax
  __int64 v15; // rdx
  signed __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 j; // rax
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 **v22; // rdi
  __int64 v23; // rax
  volatile signed __int32 *v24; // rax
  __int64 **v25; // rdi
  __int64 v26; // rax
  volatile signed __int32 *v27; // rax
  __int64 **v28; // rdi
  __int64 v29; // rax
  volatile signed __int32 *v30; // rax
  __int64 **v31; // rdi
  __int64 v32; // rax
  volatile signed __int32 *v33; // rax
  __int64 v34; // rcx
  unsigned __int8 v35; // al
  unsigned int v36; // r12d
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r13d
  unsigned int v40; // ecx
  int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  int v46; // r8d
  __int64 v47; // rax
  int v48; // r12d
  __int64 *v49; // r13
  __int64 *v50; // r12
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rdx
  __int64 v54; // rcx
  unsigned int v55; // r13d
  volatile signed __int32 *v56; // r12
  volatile signed __int32 **v57; // r15
  __int64 v58; // rax
  unsigned int v59; // ebx
  unsigned int v60; // ebx
  __int64 *v61; // rbx
  int v62; // eax
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // r14
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rbx
  __int64 v69; // rdx
  __int64 k; // r13
  __int64 **v71; // rax
  unsigned int v72; // r13d
  int v73; // r12d
  __int64 v74; // rsi
  int v75; // ebx
  char v76; // dl
  __int64 v77; // r12
  unsigned int v78; // r14d
  char v79; // r13
  char v80; // dl
  __int128 *v81; // rbx
  __int64 v82; // rdx
  int v83; // eax
  __int64 v84; // r14
  __int64 v85; // r13
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 **v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned __int8 v91; // al
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 **v94; // rax
  __int64 v95; // [rsp+0h] [rbp-300h]
  volatile signed __int32 **v96; // [rsp+8h] [rbp-2F8h]
  unsigned int v97; // [rsp+10h] [rbp-2F0h]
  __int64 v98; // [rsp+20h] [rbp-2E0h]
  unsigned int v99; // [rsp+28h] [rbp-2D8h]
  int v100; // [rsp+34h] [rbp-2CCh]
  unsigned int v101; // [rsp+34h] [rbp-2CCh]
  __int64 v102; // [rsp+38h] [rbp-2C8h]
  int v103; // [rsp+38h] [rbp-2C8h]
  unsigned int v104; // [rsp+40h] [rbp-2C0h]
  unsigned int v105; // [rsp+44h] [rbp-2BCh]
  unsigned int v106; // [rsp+48h] [rbp-2B8h]
  unsigned int v107; // [rsp+4Ch] [rbp-2B4h]
  __int64 v108; // [rsp+50h] [rbp-2B0h]
  bool v109; // [rsp+50h] [rbp-2B0h]
  unsigned __int8 v110; // [rsp+58h] [rbp-2A8h]
  __int64 v111; // [rsp+58h] [rbp-2A8h]
  __int64 v112; // [rsp+68h] [rbp-298h]
  unsigned int v115; // [rsp+7Ch] [rbp-284h]
  __int64 v116; // [rsp+88h] [rbp-278h] BYREF
  unsigned __int64 v117; // [rsp+90h] [rbp-270h] BYREF
  unsigned int v118; // [rsp+98h] [rbp-268h]
  __int64 v119; // [rsp+A0h] [rbp-260h] BYREF
  volatile signed __int32 *v120; // [rsp+A8h] [rbp-258h] BYREF
  __int64 v121; // [rsp+B0h] [rbp-250h]
  __m128i v122; // [rsp+C0h] [rbp-240h] BYREF
  _BYTE v123[560]; // [rsp+D0h] [rbp-230h] BYREF

  v115 = a2;
  if ( a2 != a3 )
  {
    sub_1526BE0(*a1, 0xBu, 4u);
    if ( a4 )
    {
      v5 = &v122;
      sub_1531130(&v119);
      v122.m128i_i8[8] |= 1u;
      v122.m128i_i64[0] = 7;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 0;
      v122.m128i_i8[8] = 6;
      sub_1525B40(v119, &v122);
      v20 = 0;
      if ( a3 )
      {
        _BitScanReverse(&v21, a3);
        v20 = 32 - (v21 ^ 0x1F);
      }
      v122.m128i_i64[0] = v20;
      v122.m128i_i8[8] = v122.m128i_i8[8] & 0xF0 | 2;
      sub_1525B40(v119, &v122);
      v22 = *a1;
      v23 = v119;
      v119 = 0;
      v122.m128i_i64[0] = v23;
      v24 = v120;
      v120 = 0;
      v122.m128i_i64[1] = (__int64)v24;
      v105 = sub_15271D0(v22, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      sub_1531130(&v122);
      sub_1526380(&v119, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      v122.m128i_i8[8] |= 1u;
      v122.m128i_i64[0] = 8;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 0;
      v122.m128i_i8[8] = 6;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 8;
      v122.m128i_i8[8] = 2;
      sub_1525B40(v119, &v122);
      v25 = *a1;
      v26 = v119;
      v119 = 0;
      v122.m128i_i64[0] = v26;
      v27 = v120;
      v120 = 0;
      v122.m128i_i64[1] = (__int64)v27;
      v107 = sub_15271D0(v25, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      sub_1531130(&v122);
      sub_1526380(&v119, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      v122.m128i_i8[8] |= 1u;
      v122.m128i_i64[0] = 9;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 0;
      v122.m128i_i8[8] = 6;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 7;
      v122.m128i_i8[8] = 2;
      sub_1525B40(v119, &v122);
      v28 = *a1;
      v29 = v119;
      v119 = 0;
      v122.m128i_i64[0] = v29;
      v30 = v120;
      v120 = 0;
      v122.m128i_i64[1] = (__int64)v30;
      v104 = sub_15271D0(v28, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      sub_1531130(&v122);
      sub_1526380(&v119, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      v122.m128i_i8[8] |= 1u;
      v122.m128i_i64[0] = 9;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 0;
      v122.m128i_i8[8] = 6;
      sub_1525B40(v119, &v122);
      v122.m128i_i64[0] = 0;
      v122.m128i_i8[8] = 8;
      sub_1525B40(v119, &v122);
      v31 = *a1;
      v32 = v119;
      v119 = 0;
      v122.m128i_i64[0] = v32;
      v33 = v120;
      v120 = 0;
      v122.m128i_i64[1] = (__int64)v33;
      v106 = sub_15271D0(v31, v122.m128i_i64);
      if ( v122.m128i_i64[1] )
        sub_A191D0((volatile signed __int32 *)v122.m128i_i64[1]);
      if ( v120 )
        sub_A191D0(v120);
    }
    else
    {
      v106 = 0;
      v5 = &v122;
      v104 = 0;
      v107 = 0;
      v105 = 0;
    }
    v112 = 0;
    v122.m128i_i64[0] = (__int64)v123;
    v122.m128i_i64[1] = 0x4000000000LL;
    while ( 1 )
    {
      v6 = (__int64)a1[17][2 * v115];
      v7 = v112;
      v8 = *(_QWORD *)v6;
      v112 = *(_QWORD *)v6;
      if ( *(_QWORD *)v6 != v7 )
      {
        v119 = (unsigned int)sub_1524C80((__int64)(a1 + 3), v8);
        sub_1525CA0((__int64)v5, &v119);
        v9 = (__int64)*a1;
        v8 = 4;
        v119 = 0x100000001LL;
        sub_152A250(v9, 4u, v122.m128i_i64[0], v122.m128i_u32[2], 0, 0, (__int64)&v119);
        v122.m128i_i32[2] = 0;
      }
      if ( *(_BYTE *)(v6 + 16) == 20 )
      {
        v119 = *(unsigned __int8 *)(v6 + 96)
             | (2 * *(unsigned __int8 *)(v6 + 97))
             | (4 * (unsigned __int8)*(_DWORD *)(v6 + 100)) & 4u;
        sub_1525CA0((__int64)v5, &v119);
        v119 = *(_QWORD *)(v6 + 32);
        sub_1525CA0((__int64)v5, &v119);
        v10 = v122.m128i_u32[2];
        v11 = *(_QWORD *)(v6 + 32);
        v12 = *(_QWORD *)(v6 + 24);
        if ( v11 > v122.m128i_u32[3] - (unsigned __int64)v122.m128i_u32[2] )
        {
          sub_16CD150(v5, v123, v11 + v122.m128i_u32[2], 8);
          v10 = v122.m128i_u32[2];
        }
        v13 = v122.m128i_i64[0] + 8 * v10;
        if ( v11 > 0 )
        {
          for ( i = 0; i != v11; ++i )
            *(_QWORD *)(v13 + 8 * i) = *(char *)(v12 + i);
          LODWORD(v10) = v122.m128i_i32[2];
        }
        v122.m128i_i32[2] = v10 + v11;
        v119 = *(_QWORD *)(v6 + 64);
        sub_1525CA0((__int64)v5, &v119);
        v15 = v122.m128i_u32[2];
        v16 = *(_QWORD *)(v6 + 64);
        v17 = *(_QWORD *)(v6 + 56);
        if ( v16 > v122.m128i_u32[3] - (unsigned __int64)v122.m128i_u32[2] )
        {
          sub_16CD150(v5, v123, v16 + v122.m128i_u32[2], 8);
          v15 = v122.m128i_u32[2];
        }
        v18 = v122.m128i_i64[0] + 8 * v15;
        if ( v16 > 0 )
        {
          for ( j = 0; j != v16; ++j )
            *(_QWORD *)(v18 + 8 * j) = *(char *)(v17 + j);
          LODWORD(v15) = v122.m128i_i32[2];
        }
        v122.m128i_i32[2] = v15 + v16;
        sub_152F3D0(*a1, 0x17u, (__int64)v5, 0);
        v122.m128i_i32[2] = 0;
        goto LABEL_21;
      }
      if ( (unsigned __int8)sub_1593BB0(v6) )
      {
        v40 = 0;
        v39 = 2;
      }
      else
      {
        v35 = *(_BYTE *)(v6 + 16);
        switch ( v35 )
        {
          case 9u:
            v40 = 0;
            v39 = 3;
            break;
          case 0xDu:
            v36 = *(_DWORD *)(v6 + 32);
            if ( v36 > 0x40 )
            {
              v46 = sub_16A57B0(v6 + 24);
              v47 = 1;
              v48 = v36 - v46;
              if ( v48 )
                v47 = ((unsigned int)(v48 - 1) >> 6) + 1;
              v49 = *(__int64 **)(v6 + 24);
              v50 = &v49[v47];
              do
              {
                while ( 1 )
                {
                  v51 = *v49;
                  if ( *v49 < 0 )
                    break;
                  ++v49;
                  v119 = 2 * v51;
                  sub_1525CA0((__int64)v5, &v119);
                  if ( v49 == v50 )
                    goto LABEL_69;
                }
                ++v49;
                v119 = -2 * v51 + 1;
                sub_1525CA0((__int64)v5, &v119);
              }
              while ( v49 != v50 );
LABEL_69:
              v40 = 0;
              v39 = 5;
            }
            else
            {
              v37 = (__int64)(*(_QWORD *)(v6 + 24) << (64 - (unsigned __int8)v36)) >> (64 - (unsigned __int8)v36);
              if ( v37 < 0 )
                v38 = -2 * v37 + 1;
              else
                v38 = 2 * v37;
              v119 = v38;
              v39 = 4;
              sub_1525CA0((__int64)v5, &v119);
              v40 = 5;
            }
            break;
          case 0xEu:
            v41 = *(unsigned __int8 *)(*(_QWORD *)v6 + 8LL);
            v42 = (unsigned int)(v41 - 1);
            if ( (unsigned __int8)(v41 - 1) > 2u )
            {
              if ( (_BYTE)v41 == 4 )
              {
                v81 = (__int128 *)&v119;
                sub_1524C40((__int64)&v119, v6 + 24, v42, v34);
                if ( (unsigned int)v120 > 0x40 )
                  v81 = (__int128 *)v119;
                v117 = *v81 >> 16;
                sub_1525CA0((__int64)v5, &v117);
                v117 = *(unsigned __int16 *)v81;
                sub_1525CA0((__int64)v5, &v117);
                if ( (unsigned int)v120 <= 0x40 )
                  goto LABEL_62;
              }
              else
              {
                if ( (unsigned __int8)(v41 - 5) > 1u )
                  goto LABEL_62;
                v61 = &v119;
                sub_1524C40((__int64)&v119, v6 + 24, v42, v34);
                if ( (unsigned int)v120 > 0x40 )
                  v61 = (__int64 *)v119;
                sub_1525CA0((__int64)v5, v61);
                sub_1525CA0((__int64)v5, v61 + 1);
                if ( (unsigned int)v120 <= 0x40 )
                  goto LABEL_62;
              }
LABEL_60:
              if ( v119 )
                j_j___libc_free_0_0(v119);
            }
            else
            {
              v43 = sub_16982C0(v6, v8, v42, v34);
              v44 = v6 + 32;
              if ( *(_QWORD *)(v6 + 32) == v43 )
                sub_169D930(&v119, v44);
              else
                sub_169D7E0(&v119, v44);
              v45 = v119;
              if ( (unsigned int)v120 > 0x40 )
                v45 = *(_QWORD *)v119;
              v117 = v45;
              sub_1525CA0((__int64)v5, &v117);
              if ( (unsigned int)v120 > 0x40 )
                goto LABEL_60;
            }
LABEL_62:
            v40 = 0;
            v39 = 6;
            break;
          default:
            v52 = v35;
            if ( (unsigned int)v35 - 11 > 1 )
              goto LABEL_96;
            if ( (unsigned __int8)sub_1595C40(v6, 8) )
            {
              v103 = sub_15958F0(v6);
              v75 = v103;
              v76 = sub_1595C70(v6);
              if ( v76 )
              {
                --v103;
                if ( v75 != 1 )
                {
                  v40 = 0;
                  v39 = 9;
LABEL_135:
                  v109 = v76;
                  v77 = v6;
                  v99 = v40;
                  v78 = 0;
                  v101 = v39;
                  v79 = v76;
                  do
                  {
                    v110 = sub_1595A50(v77, v78);
                    v119 = v110;
                    sub_1525CA0((__int64)v5, &v119);
                    v79 &= (unsigned __int8)~v110 >> 7;
                    if ( v109 && (unsigned __int8)((v110 & 0xDF) - 65) > 0x19u && (unsigned __int8)(v110 - 48) > 9u )
                      v109 = v110 == 95 || v110 == 46;
                    ++v78;
                  }
                  while ( v78 != v103 );
                  v80 = v79;
                  v40 = v99;
                  v39 = v101;
                  if ( v109 )
                  {
                    v40 = v106;
                  }
                  else if ( v80 )
                  {
                    v40 = v104;
                  }
                  break;
                }
                v40 = v106;
                v39 = 9;
              }
              else
              {
                v40 = v107;
                v39 = 8;
                if ( v103 )
                  goto LABEL_135;
              }
            }
            else
            {
              v52 = *(unsigned __int8 *)(v6 + 16);
              v35 = *(_BYTE *)(v6 + 16);
              if ( (unsigned int)(v52 - 11) > 1 )
              {
LABEL_96:
                if ( (unsigned int)(v52 - 6) <= 2 )
                {
                  v82 = sub_13CF970(v6);
                  v83 = *(_DWORD *)(v6 + 20);
                  v84 = v82;
                  v85 = v82 + 24LL * (v83 & 0xFFFFFFF);
                  if ( v82 != v85 )
                  {
                    do
                    {
                      v84 += 24;
                      v119 = (unsigned int)sub_153E840(a1 + 3);
                      sub_1525CA0((__int64)v5, &v119);
                    }
                    while ( v85 != v84 );
                  }
                  v40 = v105;
                  v39 = 7;
                }
                else if ( v35 == 5 )
                {
                  v62 = *(unsigned __int16 *)(v6 + 18);
                  v63 = (__int64)(a1 + 3);
                  switch ( (__int16)v62 )
                  {
                    case ' ':
                      v90 = sub_16348C0(v6);
                      v119 = (unsigned int)sub_1524C80(v63, v90);
                      sub_1525CA0((__int64)v5, &v119);
                      v91 = *(_BYTE *)(v6 + 17) >> 1;
                      if ( (int)v91 >> 1 )
                      {
                        v39 = 24;
                        v119 = (2 * ((int)v91 >> 1) - 2) | (unsigned int)((*(_BYTE *)(v6 + 17) & 2) != 0);
                        sub_1525CA0((__int64)v5, &v119);
                      }
                      else
                      {
                        v39 = (*(_BYTE *)(v6 + 17) & 2) == 0 ? 12 : 20;
                      }
                      v40 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
                      if ( v40 )
                      {
                        v111 = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
                        v92 = 0;
                        do
                        {
                          v93 = sub_13CF970(v6);
                          v119 = (unsigned int)sub_1524C80(v63, **(_QWORD **)(v93 + v92));
                          sub_1525CA0((__int64)v5, &v119);
                          sub_13CF970(v6);
                          v92 += 24;
                          v119 = (unsigned int)sub_153E840(v63);
                          sub_1525CA0((__int64)v5, &v119);
                        }
                        while ( v111 != v92 );
                        v40 = 0;
                      }
                      break;
                    case '3':
                    case '4':
                      v39 = 17;
                      v71 = (__int64 **)sub_13CF970(v6);
                      v119 = (unsigned int)sub_1524C80(v63, **v71);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v119 = (unsigned int)sub_1594720(v6);
                      sub_1525CA0((__int64)v5, &v119);
                      v40 = 0;
                      break;
                    case '7':
                      v39 = 13;
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v40 = 0;
                      break;
                    case ';':
                      v39 = 14;
                      v88 = (__int64 **)sub_13CF970(v6);
                      v119 = (unsigned int)sub_1524C80(v63, **v88);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v89 = sub_13CF970(v6);
                      v119 = (unsigned int)sub_1524C80(v63, **(_QWORD **)(v89 + 24));
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v40 = 0;
                      break;
                    case '<':
                      v39 = 15;
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v87 = sub_13CF970(v6);
                      v119 = (unsigned int)sub_1524C80(v63, **(_QWORD **)(v87 + 48));
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v40 = 0;
                      break;
                    case '=':
                      v86 = **(_QWORD **)sub_13CF970(v6);
                      if ( v86 == *(_QWORD *)v6 )
                      {
                        v39 = 16;
                      }
                      else
                      {
                        v39 = 19;
                        v119 = (unsigned int)sub_1524C80(v63, v86);
                        sub_1525CA0((__int64)v5, &v119);
                        sub_13CF970(v6);
                      }
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      sub_13CF970(v6);
                      v119 = (unsigned int)sub_153E840(v63);
                      sub_1525CA0((__int64)v5, &v119);
                      v40 = 0;
                      break;
                    default:
                      if ( (unsigned int)(v62 - 36) <= 0xC )
                      {
                        v39 = 11;
                        v119 = (unsigned int)(v62 - 36);
                        sub_1525CA0((__int64)v5, &v119);
                        v94 = (__int64 **)sub_13CF970(v6);
                        v119 = (unsigned int)sub_1524C80(v63, **v94);
                        sub_1525CA0((__int64)v5, &v119);
                        sub_13CF970(v6);
                        v119 = (unsigned int)sub_153E840(v63);
                        sub_1525CA0((__int64)v5, &v119);
                        v40 = 6;
                      }
                      else
                      {
                        v119 = (unsigned int)dword_4292C80[v62 - 11];
                        sub_1525CA0((__int64)v5, &v119);
                        sub_13CF970(v6);
                        v119 = (unsigned int)sub_153E840(v63);
                        sub_1525CA0((__int64)v5, &v119);
                        sub_13CF970(v6);
                        v119 = (unsigned int)sub_153E840(v63);
                        sub_1525CA0((__int64)v5, &v119);
                        v119 = sub_1523EE0(v6);
                        if ( v119 )
                          sub_1525CA0((__int64)v5, &v119);
                        v40 = 0;
                        v39 = 10;
                      }
                      break;
                  }
                }
                else
                {
                  v39 = 21;
                  v119 = (unsigned int)sub_1524C80((__int64)(a1 + 3), **(_QWORD **)(v6 - 48));
                  sub_1525CA0((__int64)v5, &v119);
                  v119 = (unsigned int)sub_153E840(a1 + 3);
                  sub_1525CA0((__int64)v5, &v119);
                  v119 = (unsigned int)sub_1548410(a1 + 3, *(_QWORD *)(v6 - 24));
                  sub_1525CA0((__int64)v5, &v119);
                  v40 = 0;
                }
                break;
              }
              if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 24LL) + 8LL) == 11 )
              {
                v72 = 0;
                v73 = sub_15958F0(v6);
                if ( v73 )
                {
                  do
                  {
                    v74 = v72++;
                    v119 = sub_1595A50(v6, v74);
                    sub_1525CA0((__int64)v5, &v119);
                  }
                  while ( v73 != v72 );
                }
              }
              else
              {
                v100 = sub_15958F0(v6);
                if ( v100 )
                {
                  v102 = (__int64)v5;
                  v55 = 0;
                  v56 = (volatile signed __int32 *)sub_16982C0(v6, 8, v53, v54);
                  v98 = v6;
                  v57 = &v120;
                  sub_1595B70(&v119, v6, 0);
                  while ( 1 )
                  {
                    if ( v120 == v56 )
                      sub_169D930(&v117, v57);
                    else
                      sub_169D7E0(&v117, v57);
                    v59 = v118;
                    if ( v118 <= 0x40 )
                    {
                      v58 = v117;
                    }
                    else
                    {
                      v60 = v59 - sub_16A57B0(&v117);
                      v58 = -1;
                      if ( v60 <= 0x40 )
                        v58 = *(_QWORD *)v117;
                    }
                    v116 = v58;
                    sub_1525CA0(v102, &v116);
                    if ( v118 > 0x40 && v117 )
                      j_j___libc_free_0_0(v117);
                    if ( v120 == v56 )
                    {
                      v108 = v121;
                      if ( v121 )
                      {
                        v64 = v121 + 32LL * *(_QWORD *)(v121 - 8);
                        if ( v121 != v64 )
                        {
                          v97 = v55;
                          v96 = v57;
                          do
                          {
                            v64 -= 32;
                            if ( v56 == *(volatile signed __int32 **)(v64 + 8) )
                            {
                              v65 = *(_QWORD *)(v64 + 16);
                              if ( v65 )
                              {
                                v66 = 32LL * *(_QWORD *)(v65 - 8);
                                v67 = v65 + v66;
                                if ( v65 != v65 + v66 )
                                {
                                  v95 = v64;
                                  do
                                  {
                                    v67 -= 32;
                                    if ( v56 == *(volatile signed __int32 **)(v67 + 8) )
                                    {
                                      v68 = *(_QWORD *)(v67 + 16);
                                      if ( v68 )
                                      {
                                        v69 = 32LL * *(_QWORD *)(v68 - 8);
                                        for ( k = v68 + v69; v68 != k; sub_127D120((_QWORD *)(k + 8)) )
                                          k -= 32;
                                        j_j_j___libc_free_0_0(v68 - 8);
                                      }
                                    }
                                    else
                                    {
                                      sub_1698460(v67 + 8);
                                    }
                                  }
                                  while ( v65 != v67 );
                                  v64 = v95;
                                }
                                j_j_j___libc_free_0_0(v65 - 8);
                              }
                            }
                            else
                            {
                              sub_1698460(v64 + 8);
                            }
                          }
                          while ( v108 != v64 );
                          v57 = v96;
                          v55 = v97;
                        }
                        j_j_j___libc_free_0_0(v108 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v57);
                    }
                    if ( v100 == ++v55 )
                      break;
                    sub_1595B70(&v119, v98, v55);
                  }
                  v5 = (__m128i *)v102;
                }
              }
              v40 = 0;
              v39 = 22;
            }
            break;
        }
      }
      sub_152F3D0(*a1, v39, (__int64)v5, v40);
      v122.m128i_i32[2] = 0;
LABEL_21:
      if ( a3 == ++v115 )
      {
        sub_15263C0(*a1);
        if ( (_BYTE *)v122.m128i_i64[0] != v123 )
          _libc_free(v122.m128i_u64[0]);
        return;
      }
    }
  }
}
