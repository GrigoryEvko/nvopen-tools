// Function: sub_2D1FFA0
// Address: 0x2d1ffa0
//
void __fastcall sub_2D1FFA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r14
  __int64 v6; // rdx
  __int64 v7; // rax
  const char **v8; // r12
  _BYTE *v9; // rbx
  const char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // r14
  _QWORD *v14; // r12
  _QWORD *j; // rbx
  _QWORD *v16; // rax
  char v17; // al
  _BYTE *v18; // rax
  int v19; // eax
  _BYTE *v20; // rdx
  char v21; // dl
  __m128i *v22; // rsi
  char v23; // al
  _QWORD *v24; // rcx
  char v25; // dl
  __m128i *v26; // rsi
  _QWORD *v27; // rcx
  char v28; // dl
  __m128i v29; // xmm7
  const char **v30; // r13
  const char *v31; // rdx
  size_t v32; // rcx
  size_t v33; // rax
  __int64 v34; // rdx
  char *v35; // rcx
  __m128i *v36; // rdi
  char v37; // dl
  _QWORD *v38; // rcx
  char v39; // dl
  __m128i *v40; // rax
  char v41; // cl
  _QWORD *v42; // rsi
  char *v43; // rax
  int v44; // eax
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i *v47; // rsi
  __m128i *v48; // rcx
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  __m128i v51; // xmm3
  __m128i v52; // xmm1
  __int64 v53; // [rsp+8h] [rbp-328h]
  __int64 v54; // [rsp+10h] [rbp-320h]
  __int64 v55; // [rsp+18h] [rbp-318h]
  __int64 v56; // [rsp+20h] [rbp-310h]
  __int64 v57; // [rsp+28h] [rbp-308h]
  __int64 v58; // [rsp+30h] [rbp-300h]
  __int64 v59; // [rsp+38h] [rbp-2F8h]
  __int64 v60; // [rsp+40h] [rbp-2F0h]
  __int64 v61; // [rsp+48h] [rbp-2E8h]
  _BYTE *v62; // [rsp+50h] [rbp-2E0h]
  char v64; // [rsp+60h] [rbp-2D0h]
  int v65; // [rsp+60h] [rbp-2D0h]
  int v66; // [rsp+60h] [rbp-2D0h]
  const char *s2; // [rsp+70h] [rbp-2C0h]
  __int64 *s2a; // [rsp+70h] [rbp-2C0h]
  size_t n; // [rsp+78h] [rbp-2B8h]
  _BYTE *na; // [rsp+78h] [rbp-2B8h]
  size_t nb; // [rsp+78h] [rbp-2B8h]
  __int64 v73; // [rsp+80h] [rbp-2B0h]
  _BYTE *v74; // [rsp+80h] [rbp-2B0h]
  __int64 *v75; // [rsp+80h] [rbp-2B0h]
  char *v76; // [rsp+88h] [rbp-2A8h]
  const char *v77; // [rsp+88h] [rbp-2A8h]
  __int64 v78[2]; // [rsp+90h] [rbp-2A0h] BYREF
  __int64 v79; // [rsp+A0h] [rbp-290h] BYREF
  __int64 v80[2]; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v81; // [rsp+C0h] [rbp-270h] BYREF
  unsigned __int64 v82[2]; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v83; // [rsp+E0h] [rbp-250h] BYREF
  __m128i v84; // [rsp+F0h] [rbp-240h] BYREF
  __m128i v85; // [rsp+100h] [rbp-230h] BYREF
  __int64 v86; // [rsp+110h] [rbp-220h]
  _QWORD v87[4]; // [rsp+120h] [rbp-210h] BYREF
  char v88; // [rsp+140h] [rbp-1F0h]
  char v89; // [rsp+141h] [rbp-1EFh]
  __m128i v90; // [rsp+150h] [rbp-1E0h] BYREF
  __m128i v91; // [rsp+160h] [rbp-1D0h] BYREF
  __int64 v92; // [rsp+170h] [rbp-1C0h]
  _QWORD v93[4]; // [rsp+180h] [rbp-1B0h] BYREF
  __int64 v94; // [rsp+1A0h] [rbp-190h]
  __m128i v95; // [rsp+1B0h] [rbp-180h] BYREF
  __m128i v96; // [rsp+1C0h] [rbp-170h] BYREF
  __int64 v97; // [rsp+1D0h] [rbp-160h]
  __m128i v98; // [rsp+1E0h] [rbp-150h] BYREF
  __m128i v99; // [rsp+1F0h] [rbp-140h] BYREF
  __int64 v100; // [rsp+200h] [rbp-130h]
  __m128i v101; // [rsp+210h] [rbp-120h] BYREF
  __m128i v102; // [rsp+220h] [rbp-110h]
  __int64 v103; // [rsp+230h] [rbp-100h]
  _QWORD v104[4]; // [rsp+240h] [rbp-F0h] BYREF
  char v105; // [rsp+260h] [rbp-D0h]
  char v106; // [rsp+261h] [rbp-CFh]
  __m128i v107; // [rsp+270h] [rbp-C0h] BYREF
  __m128i v108; // [rsp+280h] [rbp-B0h] BYREF
  __int64 v109; // [rsp+290h] [rbp-A0h]
  _QWORD v110[4]; // [rsp+2A0h] [rbp-90h] BYREF
  __int16 v111; // [rsp+2C0h] [rbp-70h]
  __m128i v112; // [rsp+2D0h] [rbp-60h] BYREF
  __m128i v113; // [rsp+2E0h] [rbp-50h] BYREF
  __int64 v114; // [rsp+2F0h] [rbp-40h]

  v3 = a2 + 72;
  *(_QWORD *)a1 = a3;
  v4 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v4 )
  {
    i = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v4 + 32);
      if ( i != v4 + 24 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        break;
      if ( !v4 )
        BUG();
    }
  }
  v64 = 0;
LABEL_8:
  while ( 1 )
  {
    v6 = i;
    if ( v3 == v4 )
      break;
    while ( 2 )
    {
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v4 + 32) )
      {
        v7 = v4 - 24;
        if ( !v4 )
          v7 = 0;
        if ( i != v7 + 48 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          break;
        if ( !v4 )
          BUG();
      }
      if ( *(_BYTE *)(v6 - 24) == 85 )
      {
        v8 = (const char **)qword_50164B0;
        v73 = v4;
        v9 = *(_BYTE **)(v6 - 56);
        do
        {
          if ( !*v9 )
          {
            if ( *v8 )
            {
              s2 = *v8;
              n = strlen(*v8);
              v10 = sub_BD5D20((__int64)v9);
              if ( n == v11 && (!n || !memcmp(v10, s2, n)) )
              {
LABEL_28:
                v64 = 1;
                v4 = v73;
                goto LABEL_8;
              }
            }
            else
            {
              sub_BD5D20((__int64)v9);
              if ( !v12 )
                goto LABEL_28;
            }
          }
          ++v8;
        }
        while ( v8 != &qword_50164B0[3] );
        v4 = v73;
        v6 = i;
        if ( v3 != v73 )
          continue;
        goto LABEL_25;
      }
      break;
    }
  }
LABEL_25:
  if ( v64 )
  {
    v13 = *(_QWORD **)(a2 + 80);
    if ( (_QWORD *)v3 != v13 )
    {
      if ( !v13 )
        BUG();
      while ( 1 )
      {
        v14 = (_QWORD *)v13[4];
        if ( v14 != v13 + 3 )
          break;
        v13 = (_QWORD *)v13[1];
        if ( (_QWORD *)v3 == v13 )
          return;
        if ( !v13 )
          BUG();
      }
      if ( v13 != (_QWORD *)v3 )
      {
        while ( 1 )
        {
          for ( j = (_QWORD *)v14[1]; ; j = (_QWORD *)v13[4] )
          {
            v16 = v13 - 3;
            if ( !v13 )
              v16 = 0;
            if ( j != v16 + 6 )
              break;
            v13 = (_QWORD *)v13[1];
            if ( (_QWORD *)v3 == v13 )
              break;
            if ( !v13 )
              BUG();
          }
          v17 = *((_BYTE *)v14 - 24);
          if ( v17 != 62 )
          {
            if ( v17 == 85 )
            {
              nb = v3;
              v30 = (const char **)&unk_50164D0;
              do
              {
                v31 = *v30;
                v32 = 0;
                if ( *v30 )
                {
                  v77 = *v30;
                  v33 = strlen(*v30);
                  v31 = v77;
                  v32 = v33;
                }
                if ( sub_2D1FC80((__int64 *)a1, (__int64)(v14 - 3), v31, v32) )
                {
                  if ( sub_2D1FC00((__int64 *)a1, v14[-4 * (*((_DWORD *)v14 - 5) & 0x7FFFFFF) - 3]) )
                  {
                    sub_B2BE50(a2);
                    sub_2D1FED0((__int64)&v112, a1, 1, 0, 0, 0, v14 + 3);
                    sub_CEB590(&v112, 1, v34, v35);
                    v36 = (__m128i *)v112.m128i_i64[0];
                    *(_BYTE *)(a1 + 8) = 1;
                    if ( v36 != &v113 )
                      j_j___libc_free_0((unsigned __int64)v36);
                  }
                }
                ++v30;
              }
              while ( v30 != (const char **)&dword_50164E8 );
              v3 = nb;
            }
            goto LABEL_45;
          }
          v18 = sub_2D1FD60((__int64 *)a1, *(v14 - 7));
          if ( !v18 )
            goto LABEL_45;
          v74 = v18;
          na = sub_2D1FE20((__int64 *)a1, (__int64)v18);
          v76 = sub_2D1FB70((__int64 *)a1, *(v14 - 11));
          if ( !v76 )
            goto LABEL_45;
          v98.m128i_i64[0] = (__int64)". Dereferencing this within the launch is undefined.";
          v62 = v74;
          s2a = v14 + 3;
          LOWORD(v100) = 259;
          v75 = (__int64 *)(v74 + 48);
          v65 = sub_B10CE0((__int64)(v14 + 3));
          v19 = sub_B10CE0((__int64)v75);
          v20 = v62;
          if ( v65 == v19
            && (v66 = sub_B10CE0((__int64)s2a), v44 = sub_B10CE0((__int64)(na + 48)), v20 = v62, v66 == v44) )
          {
            v93[0] = "was passed as a launch argument";
            LOWORD(v94) = 259;
          }
          else
          {
            sub_B2BE50(*(_QWORD *)(*((_QWORD *)v20 + 5) + 72LL));
            sub_2C75F20((__int64)&v112, v75);
            v93[0] = "was stored into the parameter buffer obtained at";
            v93[2] = &v112;
            LOWORD(v94) = 1027;
            if ( (__m128i *)v112.m128i_i64[0] != &v113 )
              j_j___libc_free_0(v112.m128i_u64[0]);
          }
          v89 = 1;
          v87[0] = " memory ";
          v88 = 3;
          if ( *v76 == 60 )
            goto LABEL_104;
          if ( *v76 != 79 )
            goto LABEL_56;
          if ( sub_CEFE10((__int64)v76) )
          {
LABEL_104:
            v43 = "local";
          }
          else
          {
            if ( !sub_CEFE40((__int64)v76) )
            {
LABEL_56:
              v84.m128i_i64[0] = (__int64)"A pointer to ";
              LOWORD(v86) = 259;
              goto LABEL_57;
            }
            v43 = "shared";
          }
          v85.m128i_i64[0] = (__int64)v43;
          v84.m128i_i64[0] = (__int64)"A pointer to ";
          LOWORD(v86) = 771;
LABEL_57:
          v21 = v88;
          if ( !v88 )
          {
            LOWORD(v92) = 256;
LABEL_84:
            LOWORD(v97) = 256;
LABEL_85:
            LOWORD(v103) = 256;
            goto LABEL_86;
          }
          if ( v88 == 1 )
          {
            v23 = v86;
            v45 = _mm_loadu_si128(&v84);
            v46 = _mm_loadu_si128(&v85);
            v92 = v86;
            v90 = v45;
            v91 = v46;
            v25 = v94;
            if ( !(_BYTE)v94 )
              goto LABEL_84;
          }
          else
          {
            v22 = &v84;
            v23 = 2;
            if ( BYTE1(v86) == 1 )
            {
              v60 = v84.m128i_i64[1];
              v22 = (__m128i *)v84.m128i_i64[0];
              v23 = 3;
            }
            if ( v89 == 1 )
            {
              v59 = v87[1];
              v24 = (_QWORD *)v87[0];
            }
            else
            {
              v24 = v87;
              v21 = 2;
            }
            v90.m128i_i64[0] = (__int64)v22;
            v91.m128i_i64[0] = (__int64)v24;
            BYTE1(v92) = v21;
            v25 = v94;
            v90.m128i_i64[1] = v60;
            v91.m128i_i64[1] = v59;
            LOBYTE(v92) = v23;
            if ( !(_BYTE)v94 )
              goto LABEL_84;
          }
          if ( v25 == 1 )
          {
            v23 = v92;
            v49 = _mm_loadu_si128(&v90);
            v50 = _mm_loadu_si128(&v91);
            v97 = v92;
            v95 = v49;
            v96 = v50;
            if ( !(_BYTE)v92 )
              goto LABEL_85;
          }
          else
          {
            if ( BYTE1(v92) == 1 )
            {
              v58 = v90.m128i_i64[1];
              v26 = (__m128i *)v90.m128i_i64[0];
            }
            else
            {
              v26 = &v90;
              v23 = 2;
            }
            if ( BYTE1(v94) == 1 )
            {
              v57 = v93[1];
              v27 = (_QWORD *)v93[0];
            }
            else
            {
              v27 = v93;
              v25 = 2;
            }
            v95.m128i_i64[0] = (__int64)v26;
            v96.m128i_i64[0] = (__int64)v27;
            v95.m128i_i64[1] = v58;
            v96.m128i_i64[1] = v57;
            LOBYTE(v97) = v23;
            BYTE1(v97) = v25;
          }
          v28 = v100;
          if ( !(_BYTE)v100 )
            goto LABEL_85;
          if ( v23 == 1 )
          {
            v29 = _mm_loadu_si128(&v99);
            v101 = _mm_loadu_si128(&v98);
            v103 = v100;
            v102 = v29;
          }
          else if ( (_BYTE)v100 == 1 )
          {
            v52 = _mm_loadu_si128(&v96);
            v101 = _mm_loadu_si128(&v95);
            v103 = v97;
            v102 = v52;
          }
          else
          {
            if ( BYTE1(v97) == 1 )
            {
              v54 = v95.m128i_i64[1];
              v47 = (__m128i *)v95.m128i_i64[0];
            }
            else
            {
              v47 = &v95;
              v23 = 2;
            }
            if ( BYTE1(v100) == 1 )
            {
              v53 = v98.m128i_i64[1];
              v48 = (__m128i *)v98.m128i_i64[0];
            }
            else
            {
              v48 = &v98;
              v28 = 2;
            }
            v101.m128i_i64[0] = (__int64)v47;
            v102.m128i_i64[0] = (__int64)v48;
            v101.m128i_i64[1] = v54;
            v102.m128i_i64[1] = v53;
            LOBYTE(v103) = v23;
            BYTE1(v103) = v28;
          }
LABEL_86:
          sub_CA0F50(v80, (void **)&v101);
          v106 = 1;
          v111 = 260;
          v110[0] = v80;
          v104[0] = " : Warning: ";
          v105 = 3;
          sub_B2BE50(*(_QWORD *)(v14[2] + 72LL));
          sub_2C75F20((__int64)v82, s2a);
          v37 = v105;
          if ( v105 )
          {
            if ( v105 == 1 )
            {
              v39 = v111;
              v107.m128i_i64[0] = (__int64)v82;
              LOWORD(v109) = 260;
              if ( (_BYTE)v111 )
              {
                if ( (_BYTE)v111 != 1 )
                {
                  v41 = 4;
                  v56 = v107.m128i_i64[1];
                  v40 = (__m128i *)v82;
                  goto LABEL_93;
                }
                goto LABEL_122;
              }
            }
            else
            {
              if ( v106 == 1 )
              {
                v38 = (_QWORD *)v104[0];
                v61 = v104[1];
              }
              else
              {
                v38 = v104;
                v37 = 2;
              }
              BYTE1(v109) = v37;
              v39 = v111;
              v107.m128i_i64[0] = (__int64)v82;
              v108.m128i_i64[0] = (__int64)v38;
              v108.m128i_i64[1] = v61;
              LOBYTE(v109) = 4;
              if ( (_BYTE)v111 )
              {
                if ( (_BYTE)v111 != 1 )
                {
                  v40 = &v107;
                  v41 = 2;
LABEL_93:
                  if ( HIBYTE(v111) == 1 )
                  {
                    v55 = v110[1];
                    v42 = (_QWORD *)v110[0];
                  }
                  else
                  {
                    v42 = v110;
                    v39 = 2;
                  }
                  v112.m128i_i64[0] = (__int64)v40;
                  v113.m128i_i64[0] = (__int64)v42;
                  v112.m128i_i64[1] = v56;
                  LOBYTE(v114) = v41;
                  v113.m128i_i64[1] = v55;
                  BYTE1(v114) = v39;
                  goto LABEL_98;
                }
LABEL_122:
                v51 = _mm_loadu_si128(&v108);
                v112 = _mm_loadu_si128(&v107);
                v114 = v109;
                v113 = v51;
                goto LABEL_98;
              }
            }
          }
          else
          {
            LOWORD(v109) = 256;
          }
          LOWORD(v114) = 256;
LABEL_98:
          sub_CA0F50(v78, (void **)&v112);
          if ( (__int64 *)v82[0] != &v83 )
            j_j___libc_free_0(v82[0]);
          if ( (__int64 *)v80[0] != &v81 )
            j_j___libc_free_0(v80[0]);
          sub_CEB650(v78);
          if ( (__int64 *)v78[0] != &v79 )
            j_j___libc_free_0(v78[0]);
LABEL_45:
          if ( (_QWORD *)v3 == v13 )
            return;
          v14 = j;
        }
      }
    }
  }
}
