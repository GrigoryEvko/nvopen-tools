// Function: sub_36D31C0
// Address: 0x36d31c0
//
__int64 __fastcall sub_36D31C0(__int64 a1, __int64 a2, char a3)
{
  unsigned int v3; // r8d
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  const char *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned __int64 *v12; // r9
  unsigned int v13; // eax
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  int v18; // eax
  _BYTE *v19; // r8
  unsigned __int64 v20; // rsi
  unsigned int v21; // edi
  unsigned __int64 v22; // r10
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // r10
  __int64 v26; // rax
  __m128i *v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rcx
  int v32; // eax
  _BYTE *v33; // rdi
  int v34; // esi
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  char v37; // r9
  __int64 v38; // r8
  __m128i v39; // rax
  char v40; // al
  _QWORD *v41; // rcx
  char v42; // dl
  char v43; // al
  __m128i v44; // xmm1
  __m128i v45; // xmm2
  char v46; // dl
  __m128i *v47; // rsi
  __m128i *v48; // rcx
  char v49; // dl
  __m128i v50; // xmm7
  __m128i v51; // xmm5
  char *v52; // rax
  char *i; // rcx
  char v54; // dl
  _QWORD *v55; // rdx
  _QWORD *v56; // rax
  __int64 v57; // r14
  char v58; // dl
  unsigned __int64 v60; // rax
  __m128i *v61; // rsi
  __int8 v62; // dl
  __m128i *v63; // rsi
  __m128i *v64; // rcx
  __m128i v65; // xmm3
  __m128i v66; // xmm4
  __m128i v67; // xmm5
  __m128i v68; // xmm6
  __m128i *v69; // rcx
  _QWORD *v70; // rsi
  __m128i v71; // xmm4
  __m128i v72; // xmm7
  size_t v73; // [rsp+8h] [rbp-388h]
  __int64 v74; // [rsp+18h] [rbp-378h]
  __int64 v75; // [rsp+20h] [rbp-370h]
  __int64 v76; // [rsp+28h] [rbp-368h]
  __int64 v77; // [rsp+38h] [rbp-358h]
  __int64 v78; // [rsp+40h] [rbp-350h]
  __int64 v79; // [rsp+48h] [rbp-348h]
  __int64 v80; // [rsp+50h] [rbp-340h]
  __int64 v81; // [rsp+58h] [rbp-338h]
  __int64 v83; // [rsp+68h] [rbp-328h]
  _QWORD *v84; // [rsp+70h] [rbp-320h]
  int *v85; // [rsp+70h] [rbp-320h]
  __int64 v87; // [rsp+B8h] [rbp-2D8h]
  unsigned __int64 v88[2]; // [rsp+C0h] [rbp-2D0h] BYREF
  __m128i v89; // [rsp+D0h] [rbp-2C0h] BYREF
  __int64 v90[2]; // [rsp+E0h] [rbp-2B0h] BYREF
  _QWORD v91[2]; // [rsp+F0h] [rbp-2A0h] BYREF
  char *v92; // [rsp+100h] [rbp-290h] BYREF
  __int64 v93; // [rsp+108h] [rbp-288h]
  __int64 v94; // [rsp+110h] [rbp-280h] BYREF
  _QWORD *v95; // [rsp+120h] [rbp-270h] BYREF
  int v96; // [rsp+128h] [rbp-268h]
  _QWORD v97[2]; // [rsp+130h] [rbp-260h] BYREF
  __m128i v98; // [rsp+140h] [rbp-250h] BYREF
  __m128i v99; // [rsp+150h] [rbp-240h] BYREF
  __int64 v100; // [rsp+160h] [rbp-230h]
  _QWORD v101[4]; // [rsp+170h] [rbp-220h] BYREF
  char v102; // [rsp+190h] [rbp-200h]
  char v103; // [rsp+191h] [rbp-1FFh]
  __m128i v104; // [rsp+1A0h] [rbp-1F0h] BYREF
  __m128i v105; // [rsp+1B0h] [rbp-1E0h] BYREF
  __int64 v106; // [rsp+1C0h] [rbp-1D0h]
  _QWORD v107[4]; // [rsp+1D0h] [rbp-1C0h] BYREF
  __int16 v108; // [rsp+1F0h] [rbp-1A0h]
  __m128i v109; // [rsp+200h] [rbp-190h] BYREF
  __m128i v110; // [rsp+210h] [rbp-180h] BYREF
  __int64 v111; // [rsp+220h] [rbp-170h]
  __m128i v112; // [rsp+230h] [rbp-160h] BYREF
  __m128i v113; // [rsp+240h] [rbp-150h] BYREF
  __int64 v114; // [rsp+250h] [rbp-140h]
  __m128i v115; // [rsp+260h] [rbp-130h] BYREF
  __m128i v116; // [rsp+270h] [rbp-120h] BYREF
  __int64 v117; // [rsp+280h] [rbp-110h]
  __m128i v118; // [rsp+290h] [rbp-100h] BYREF
  __m128i v119; // [rsp+2A0h] [rbp-F0h] BYREF
  __int64 v120; // [rsp+2B0h] [rbp-E0h]
  __m128i v121; // [rsp+2C0h] [rbp-D0h] BYREF
  __m128i v122; // [rsp+2D0h] [rbp-C0h] BYREF
  __int64 v123; // [rsp+2E0h] [rbp-B0h]

  v3 = 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v4 == 9 && (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 0 )
  {
    v5 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    {
      v6 = *(_QWORD *)(v4 - 8);
      v76 = v6 + v5;
    }
    else
    {
      v76 = *(_QWORD *)(a2 - 32);
      v6 = v4 - v5;
    }
    v7 = "__fini_array_object_";
    v87 = v6;
    if ( a3 )
      v7 = "__init_array_object_";
    v81 = (__int64)v7;
    while ( 1 )
    {
      v8 = *(_DWORD *)(*(_QWORD *)v87 + 4LL) & 0x7FFFFFF;
      v9 = 32 * (1 - v8);
      v10 = *(_QWORD *)(*(_QWORD *)v87 - 32 * v8);
      v11 = *(_QWORD *)(*(_QWORD *)v87 + v9);
      v12 = *(unsigned __int64 **)(v10 + 24);
      v13 = *(_DWORD *)(v10 + 32);
      if ( v13 > 0x40 )
      {
        v14 = *v12;
      }
      else
      {
        if ( !v13 )
        {
          v14 = 0;
          v121.m128i_i64[0] = (__int64)&v122;
          sub_2240A50(v121.m128i_i64, 1u, 0);
          v19 = (_BYTE *)v121.m128i_i64[0];
          LOBYTE(v20) = 0;
LABEL_24:
          *v19 = v20 + 48;
          goto LABEL_25;
        }
        v14 = (__int64)((_QWORD)v12 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
      }
      if ( v14 <= 9 )
      {
        v121.m128i_i64[0] = (__int64)&v122;
        sub_2240A50(v121.m128i_i64, 1u, 0);
        v19 = (_BYTE *)v121.m128i_i64[0];
        LOBYTE(v20) = v14;
        goto LABEL_24;
      }
      if ( v14 <= 0x63 )
      {
        v121.m128i_i64[0] = (__int64)&v122;
        sub_2240A50(v121.m128i_i64, 2u, 0);
        v19 = (_BYTE *)v121.m128i_i64[0];
        v20 = v14;
      }
      else
      {
        if ( v14 <= 0x3E7 )
        {
          v16 = 3;
        }
        else if ( v14 <= 0x270F )
        {
          v16 = 4;
        }
        else
        {
          v15 = v14;
          LODWORD(v16) = 1;
          while ( 1 )
          {
            v17 = v15;
            v18 = v16;
            v16 = (unsigned int)(v16 + 4);
            v15 /= 0x2710u;
            if ( v17 <= 0x1869F )
              break;
            if ( v17 <= 0xF423F )
            {
              v16 = (unsigned int)(v18 + 5);
              v121.m128i_i64[0] = (__int64)&v122;
              goto LABEL_21;
            }
            if ( v17 <= (unsigned __int64)&loc_98967F )
            {
              v16 = (unsigned int)(v18 + 6);
              break;
            }
            if ( v17 <= 0x5F5E0FF )
            {
              v16 = (unsigned int)(v18 + 7);
              break;
            }
          }
        }
        v121.m128i_i64[0] = (__int64)&v122;
LABEL_21:
        sub_2240A50(v121.m128i_i64, v16, 0);
        v19 = (_BYTE *)v121.m128i_i64[0];
        v20 = v14;
        v21 = v121.m128i_i32[2] - 1;
        do
        {
          v22 = v20;
          v23 = 5
              * (v20 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v20 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v24 = v20;
          v20 /= 0x64u;
          v25 = v22 - 4 * v23;
          v19[v21] = a00010203040506_0[2 * v25 + 1];
          v26 = v21 - 1;
          v21 -= 2;
          v19[v26] = a00010203040506_0[2 * v25];
        }
        while ( v24 > 0x270F );
        if ( v24 <= 0x3E7 )
          goto LABEL_24;
      }
      v19[1] = a00010203040506_0[2 * v20 + 1];
      *v19 = a00010203040506_0[2 * v20];
LABEL_25:
      v27 = (__m128i *)sub_2241130((unsigned __int64 *)&v121, 0, 0, ".", 1u);
      v88[0] = (unsigned __int64)&v89;
      if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
      {
        v89 = _mm_loadu_si128(v27 + 1);
      }
      else
      {
        v88[0] = v27->m128i_i64[0];
        v89.m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v28 = v27->m128i_u64[1];
      v27[1].m128i_i8[0] = 0;
      v88[1] = v28;
      v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
      v27->m128i_i64[1] = 0;
      if ( (__m128i *)v121.m128i_i64[0] != &v122 )
        j_j___libc_free_0(v121.m128i_u64[0]);
      if ( qword_5040D50 )
      {
        v90[0] = (__int64)v91;
        sub_36D16D0(v90, (_BYTE *)qword_5040D48, qword_5040D48 + qword_5040D50);
      }
      else
      {
        v73 = *(_QWORD *)(a1 + 208);
        v85 = *(int **)(a1 + 200);
        sub_C7D030(&v121);
        sub_C7D280(v121.m128i_i32, v85, v73);
        sub_C7D290(&v121, &v115);
        v60 = v115.m128i_i64[0];
        if ( v115.m128i_i64[0] )
        {
          v61 = (__m128i *)&v119.m128i_i8[1];
          do
          {
            v61 = (__m128i *)((char *)v61 - 1);
            v62 = a0123456789abcd_10[v60 & 0xF] | 0x20;
            v60 >>= 4;
            v61->m128i_i8[0] = v62;
          }
          while ( v60 );
        }
        else
        {
          v119.m128i_i8[0] = 48;
          v61 = &v119;
        }
        v90[0] = (__int64)v91;
        sub_36D16D0(v90, v61, (__int64)v119.m128i_i64 + 1);
      }
      if ( v14 <= 9 )
      {
        v95 = v97;
        sub_2240A50((__int64 *)&v95, 1u, 0);
        v33 = v95;
LABEL_44:
        *v33 = v14 + 48;
        goto LABEL_45;
      }
      if ( v14 <= 0x63 )
      {
        v95 = v97;
        sub_2240A50((__int64 *)&v95, 2u, 0);
        v33 = v95;
      }
      else
      {
        if ( v14 <= 0x3E7 )
        {
          v30 = 3;
        }
        else if ( v14 <= 0x270F )
        {
          v30 = 4;
        }
        else
        {
          v29 = v14;
          LODWORD(v30) = 1;
          while ( 1 )
          {
            v31 = v29;
            v32 = v30;
            v30 = (unsigned int)(v30 + 4);
            v29 /= 0x2710u;
            if ( v31 <= 0x1869F )
              break;
            if ( v31 <= 0xF423F )
            {
              v30 = (unsigned int)(v32 + 5);
              v95 = v97;
              goto LABEL_41;
            }
            if ( v31 <= (unsigned __int64)&loc_98967F )
            {
              v30 = (unsigned int)(v32 + 6);
              break;
            }
            if ( v31 <= 0x5F5E0FF )
            {
              v30 = (unsigned int)(v32 + 7);
              break;
            }
          }
        }
        v95 = v97;
LABEL_41:
        sub_2240A50((__int64 *)&v95, v30, 0);
        v33 = v95;
        v34 = v96 - 1;
        do
        {
          v35 = v14
              - 20
              * (v14 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v14 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v36 = v14;
          v14 /= 0x64u;
          v37 = a00010203040506_0[2 * v35 + 1];
          LOBYTE(v35) = a00010203040506_0[2 * v35];
          v33[v34] = v37;
          v38 = (unsigned int)(v34 - 1);
          v34 -= 2;
          v33[v38] = v35;
        }
        while ( v36 > 0x270F );
        if ( v36 <= 0x3E7 )
          goto LABEL_44;
      }
      v33[1] = a00010203040506_0[2 * v14 + 1];
      *v33 = a00010203040506_0[2 * v14];
LABEL_45:
      LOWORD(v120) = 260;
      v118.m128i_i64[0] = (__int64)&v95;
      v107[0] = v90;
      v112.m128i_i64[0] = (__int64)"_";
      LOWORD(v114) = 259;
      v108 = 260;
      v103 = 1;
      v101[0] = "_";
      v102 = 3;
      v39.m128i_i64[0] = (__int64)sub_BD5D20(v11);
      v99 = v39;
      v40 = v102;
      LOWORD(v100) = 1283;
      v98.m128i_i64[0] = v81;
      if ( !v102 )
      {
        LOWORD(v106) = 256;
LABEL_88:
        LOWORD(v111) = 256;
LABEL_89:
        LOWORD(v117) = 256;
        goto LABEL_90;
      }
      if ( v102 == 1 )
      {
        v72 = _mm_load_si128(&v99);
        v42 = v108;
        v104 = _mm_load_si128(&v98);
        v106 = v100;
        v105 = v72;
        if ( !(_BYTE)v108 )
          goto LABEL_88;
        if ( (_BYTE)v108 == 1 )
        {
LABEL_51:
          v43 = v106;
          v44 = _mm_load_si128(&v104);
          v45 = _mm_load_si128(&v105);
          v111 = v106;
          v109 = v44;
          v110 = v45;
          if ( (_BYTE)v106 )
          {
            v46 = v114;
            if ( (_BYTE)v114 )
            {
              if ( (_BYTE)v106 == 1 )
              {
                v67 = _mm_load_si128(&v112);
                v68 = _mm_load_si128(&v113);
                v117 = v114;
                v43 = v114;
                v115 = v67;
                v116 = v68;
              }
              else
              {
LABEL_54:
                if ( v46 == 1 )
                {
                  v43 = v111;
                  v65 = _mm_load_si128(&v109);
                  v66 = _mm_load_si128(&v110);
                  v117 = v111;
                  v115 = v65;
                  v116 = v66;
                  if ( !(_BYTE)v111 )
                  {
                    LOWORD(v123) = 256;
                    goto LABEL_63;
                  }
                }
                else
                {
                  if ( BYTE1(v111) == 1 )
                  {
                    v80 = v109.m128i_i64[1];
                    v47 = (__m128i *)v109.m128i_i64[0];
                  }
                  else
                  {
                    v47 = &v109;
                    v43 = 2;
                  }
                  if ( BYTE1(v114) == 1 )
                  {
                    v79 = v112.m128i_i64[1];
                    v48 = (__m128i *)v112.m128i_i64[0];
                  }
                  else
                  {
                    v48 = &v112;
                    v46 = 2;
                  }
                  v115.m128i_i64[0] = (__int64)v47;
                  v116.m128i_i64[0] = (__int64)v48;
                  v115.m128i_i64[1] = v80;
                  LOBYTE(v117) = v43;
                  v116.m128i_i64[1] = v79;
                  BYTE1(v117) = v46;
                }
              }
              v49 = v120;
              if ( (_BYTE)v120 )
              {
                if ( v43 == 1 )
                {
                  v50 = _mm_load_si128(&v118);
                  v51 = _mm_load_si128(&v119);
                  v123 = v120;
                  v121 = v50;
                  v122 = v51;
                }
                else if ( (_BYTE)v120 == 1 )
                {
                  v71 = _mm_load_si128(&v116);
                  v121 = _mm_load_si128(&v115);
                  v123 = v117;
                  v122 = v71;
                }
                else
                {
                  if ( BYTE1(v117) == 1 )
                  {
                    v78 = v115.m128i_i64[1];
                    v63 = (__m128i *)v115.m128i_i64[0];
                  }
                  else
                  {
                    v63 = &v115;
                    v43 = 2;
                  }
                  if ( BYTE1(v120) == 1 )
                  {
                    v77 = v118.m128i_i64[1];
                    v64 = (__m128i *)v118.m128i_i64[0];
                  }
                  else
                  {
                    v64 = &v118;
                    v49 = 2;
                  }
                  v121.m128i_i64[0] = (__int64)v63;
                  v122.m128i_i64[0] = (__int64)v64;
                  v121.m128i_i64[1] = v78;
                  LOBYTE(v123) = v43;
                  v122.m128i_i64[1] = v77;
                  BYTE1(v123) = v49;
                }
                goto LABEL_63;
              }
              goto LABEL_90;
            }
          }
          goto LABEL_89;
        }
        if ( BYTE1(v106) == 1 )
        {
          v75 = v104.m128i_i64[1];
          v69 = (__m128i *)v104.m128i_i64[0];
          v43 = 3;
          goto LABEL_120;
        }
      }
      else
      {
        if ( v103 == 1 )
        {
          v41 = (_QWORD *)v101[0];
          v83 = v101[1];
        }
        else
        {
          v41 = v101;
          v40 = 2;
        }
        v105.m128i_i64[0] = (__int64)v41;
        v104.m128i_i64[0] = (__int64)&v98;
        v42 = v108;
        v105.m128i_i64[1] = v83;
        LOBYTE(v106) = 2;
        BYTE1(v106) = v40;
        if ( !(_BYTE)v108 )
          goto LABEL_88;
        if ( (_BYTE)v108 == 1 )
          goto LABEL_51;
      }
      v69 = &v104;
      v43 = 2;
LABEL_120:
      if ( HIBYTE(v108) == 1 )
      {
        v70 = (_QWORD *)v107[0];
        v74 = v107[1];
      }
      else
      {
        v70 = v107;
        v42 = 2;
      }
      BYTE1(v111) = v42;
      v46 = v114;
      v109.m128i_i64[0] = (__int64)v69;
      v109.m128i_i64[1] = v75;
      v110.m128i_i64[0] = (__int64)v70;
      v110.m128i_i64[1] = v74;
      LOBYTE(v111) = v43;
      if ( (_BYTE)v114 )
        goto LABEL_54;
      LOWORD(v117) = 256;
LABEL_90:
      LOWORD(v123) = 256;
LABEL_63:
      sub_CA0F50((__int64 *)&v92, (void **)&v121);
      if ( v95 != v97 )
        j_j___libc_free_0((unsigned __int64)v95);
      v52 = v92;
      for ( i = &v92[v93]; v52 != i; *(v52 - 1) = v54 )
      {
        v54 = *v52;
        if ( *v52 == 46 )
          v54 = 95;
        ++v52;
      }
      v55 = *(_QWORD **)(v11 + 8);
      LOWORD(v123) = 260;
      v84 = v55;
      v121.m128i_i64[0] = (__int64)&v92;
      v118.m128i_i64[0] = 0x100000004LL;
      v56 = sub_BD2C40(88, unk_3F0FAE8);
      v57 = (__int64)v56;
      if ( v56 )
        sub_B30000((__int64)v56, a1, v84, 1, 0, v11, (__int64)&v121, 0, 0, v118.m128i_i64[0], 0);
      if ( a3 )
        sub_8FD6D0((__int64)&v121, ".init_array", v88);
      else
        sub_8FD6D0((__int64)&v121, ".fini_array", v88);
      sub_B31A00(v57, v121.m128i_i64[0], v121.m128i_i64[1]);
      if ( (__m128i *)v121.m128i_i64[0] != &v122 )
        j_j___libc_free_0(v121.m128i_u64[0]);
      v58 = *(_BYTE *)(v57 + 32) & 0xCF | 0x20;
      *(_BYTE *)(v57 + 32) = v58;
      if ( (v58 & 0xF) != 9 )
        *(_BYTE *)(v57 + 33) |= 0x40u;
      v121.m128i_i64[0] = v57;
      sub_2A413E0((__int64 **)a1, (unsigned __int64 *)&v121, 1);
      if ( v92 != (char *)&v94 )
        j_j___libc_free_0((unsigned __int64)v92);
      if ( (_QWORD *)v90[0] != v91 )
        j_j___libc_free_0(v90[0]);
      if ( (__m128i *)v88[0] != &v89 )
        j_j___libc_free_0(v88[0]);
      v87 += 32;
      if ( v76 == v87 )
        return 1;
    }
  }
  return v3;
}
