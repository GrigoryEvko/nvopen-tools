// Function: sub_1CEC2A0
// Address: 0x1cec2a0
//
void __fastcall sub_1CEC2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 i; // r14
  __int64 v6; // rdx
  __int64 v7; // rax
  const char **v8; // r13
  __int64 v9; // r8
  const char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  _QWORD *v14; // r13
  _QWORD *j; // r15
  _QWORD *v16; // rax
  char v17; // al
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rdx
  char v21; // al
  char v22; // dl
  char *v23; // rax
  char v24; // al
  const char **v25; // rdx
  char v26; // al
  __m128i *v27; // rsi
  char v28; // dl
  __int64 **v29; // rcx
  const char **v30; // rbx
  const char *v31; // rdx
  size_t v32; // rcx
  size_t v33; // rax
  _QWORD *v34; // rdi
  __m128i *v35; // rsi
  char v36; // al
  const char **v37; // rcx
  char v38; // dl
  __m128i *v39; // rsi
  _QWORD *v40; // rcx
  char v41; // dl
  __m128i *v42; // rsi
  __m128i *v43; // rcx
  int v44; // eax
  bool v45; // al
  __m128i v46; // xmm2
  __int64 v47; // [rsp+0h] [rbp-230h]
  char v49; // [rsp+10h] [rbp-220h]
  int v50; // [rsp+10h] [rbp-220h]
  int v51; // [rsp+10h] [rbp-220h]
  const char *s2; // [rsp+20h] [rbp-210h]
  __int64 *s2a; // [rsp+20h] [rbp-210h]
  size_t n; // [rsp+28h] [rbp-208h]
  _QWORD *na; // [rsp+28h] [rbp-208h]
  size_t nb; // [rsp+28h] [rbp-208h]
  __int64 v58; // [rsp+30h] [rbp-200h]
  __int64 v59; // [rsp+30h] [rbp-200h]
  __int64 v60; // [rsp+30h] [rbp-200h]
  __int64 *v61; // [rsp+30h] [rbp-200h]
  __int64 v62; // [rsp+38h] [rbp-1F8h]
  const char *v63; // [rsp+38h] [rbp-1F8h]
  __m128i v64; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v65; // [rsp+50h] [rbp-1E0h]
  const char *v66; // [rsp+60h] [rbp-1D0h] BYREF
  char v67; // [rsp+70h] [rbp-1C0h]
  char v68; // [rsp+71h] [rbp-1BFh]
  __m128i v69; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v70; // [rsp+90h] [rbp-1A0h]
  _QWORD v71[2]; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v72; // [rsp+B0h] [rbp-180h]
  __m128i v73; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v74; // [rsp+D0h] [rbp-160h]
  __m128i v75; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v76; // [rsp+F0h] [rbp-140h]
  __m128i v77; // [rsp+100h] [rbp-130h] BYREF
  __int64 v78; // [rsp+110h] [rbp-120h]
  const char *v79; // [rsp+120h] [rbp-110h] BYREF
  char v80; // [rsp+130h] [rbp-100h]
  char v81; // [rsp+131h] [rbp-FFh]
  __m128i v82; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v83; // [rsp+150h] [rbp-E0h]
  __int64 *v84; // [rsp+160h] [rbp-D0h] BYREF
  __int16 v85; // [rsp+170h] [rbp-C0h]
  __m128i v86; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+190h] [rbp-A0h]
  __int64 v88[2]; // [rsp+1A0h] [rbp-90h] BYREF
  __int64 v89; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v90[2]; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v91; // [rsp+1D0h] [rbp-60h] BYREF
  _QWORD v92[2]; // [rsp+1E0h] [rbp-50h] BYREF
  _QWORD v93[8]; // [rsp+1F0h] [rbp-40h] BYREF

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
      i = *(_QWORD *)(v4 + 24);
      if ( i != v4 + 16 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        break;
      if ( !v4 )
        BUG();
    }
  }
  v49 = 0;
LABEL_8:
  while ( 1 )
  {
    v6 = i;
    if ( v3 == v4 )
      break;
    while ( 2 )
    {
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v4 + 24) )
      {
        v7 = v4 - 24;
        if ( !v4 )
          v7 = 0;
        if ( i != v7 + 40 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          break;
        if ( !v4 )
          BUG();
      }
      if ( *(_BYTE *)(v6 - 8) == 78 )
      {
        v8 = (const char **)qword_4FC0600;
        v9 = *(_QWORD *)(v6 - 48);
        do
        {
          if ( !*(_BYTE *)(v9 + 16) )
          {
            if ( *v8 )
            {
              s2 = *v8;
              v58 = v9;
              n = strlen(*v8);
              v10 = sub_1649960(v58);
              v9 = v58;
              if ( n == v11 )
              {
                if ( !n || !memcmp(v10, s2, n) )
                {
LABEL_26:
                  v49 = 1;
                  goto LABEL_8;
                }
                v9 = v58;
              }
            }
            else
            {
              v59 = v9;
              sub_1649960(v9);
              v9 = v59;
              if ( !v12 )
                goto LABEL_26;
            }
          }
          ++v8;
        }
        while ( v8 != &qword_4FC0600[3] );
        v6 = i;
        if ( v3 != v4 )
          continue;
        goto LABEL_23;
      }
      break;
    }
  }
LABEL_23:
  if ( v49 )
  {
    v13 = *(_QWORD **)(a2 + 80);
    if ( (_QWORD *)v3 != v13 )
    {
      if ( !v13 )
        BUG();
      while ( 1 )
      {
        v14 = (_QWORD *)v13[3];
        if ( v14 != v13 + 2 )
          break;
        v13 = (_QWORD *)v13[1];
        if ( (_QWORD *)v3 == v13 )
          return;
        if ( !v13 )
          BUG();
      }
      if ( (_QWORD *)v3 != v13 )
      {
        while ( 1 )
        {
          for ( j = (_QWORD *)v14[1]; ; j = (_QWORD *)v13[3] )
          {
            v16 = v13 - 3;
            if ( !v13 )
              v16 = 0;
            if ( j != v16 + 5 )
              break;
            v13 = (_QWORD *)v13[1];
            if ( (_QWORD *)v3 == v13 )
              break;
            if ( !v13 )
              BUG();
          }
          v17 = *((_BYTE *)v14 - 8);
          if ( v17 != 55 )
          {
            if ( v17 == 78 )
            {
              nb = (size_t)v13;
              v30 = (const char **)&unk_4FC0620;
              do
              {
                v31 = *v30;
                v32 = 0;
                if ( *v30 )
                {
                  v63 = *v30;
                  v33 = strlen(*v30);
                  v31 = v63;
                  v32 = v33;
                }
                if ( sub_1CEBF90((__int64 *)a1, (__int64)(v14 - 3), v31, v32) )
                {
                  if ( sub_1CEBF10((__int64 *)a1, (__int64 *)v14[-3 * (*((_DWORD *)v14 - 1) & 0xFFFFFFF) - 3]) )
                  {
                    sub_15E0530(a2);
                    sub_1CEC1E0((__int64)v92, a1, 1, 0, 0, 0, v14 + 3);
                    sub_1C3EFD0((__int64)v92, 1);
                    v34 = (_QWORD *)v92[0];
                    *(_BYTE *)(a1 + 8) = 1;
                    if ( v34 != v93 )
                      j_j___libc_free_0(v34, v93[0] + 1LL);
                  }
                }
                ++v30;
              }
              while ( &dword_4FC0638 != (int *)v30 );
              v13 = (_QWORD *)nb;
            }
            goto LABEL_46;
          }
          v18 = sub_1CEC070((__int64 *)a1, (__int64 *)*(v14 - 6));
          if ( !v18 )
            goto LABEL_46;
          v60 = v18;
          na = sub_1CEC130((__int64 *)a1, v18);
          v62 = sub_1CEBE70((__int64 *)a1, (__int64 *)*(v14 - 9));
          if ( !v62 )
            goto LABEL_46;
          v75.m128i_i64[0] = (__int64)". Dereferencing this within the launch is undefined.";
          v47 = v60;
          s2a = v14 + 3;
          LOWORD(v76) = 259;
          v61 = (__int64 *)(v60 + 48);
          v50 = sub_15C70B0((__int64)(v14 + 3));
          v19 = sub_15C70B0((__int64)v61);
          v20 = v47;
          if ( v50 == v19
            && (v51 = sub_15C70B0((__int64)s2a), v44 = sub_15C70B0((__int64)(na + 6)), v20 = v47, v51 == v44) )
          {
            v71[0] = "was passed as a launch argument";
            LOWORD(v72) = 259;
          }
          else
          {
            sub_15E0530(*(_QWORD *)(*(_QWORD *)(v20 + 40) + 56LL));
            sub_1C315E0((__int64)v92, v61);
            v71[0] = "was stored into the parameter buffer obtained at";
            v71[1] = v92;
            LOWORD(v72) = 1027;
            if ( (_QWORD *)v92[0] != v93 )
              j_j___libc_free_0(v92[0], v93[0] + 1LL);
          }
          v68 = 1;
          v66 = " memory ";
          v67 = 3;
          v21 = *(_BYTE *)(v62 + 16);
          if ( v21 == 53 )
          {
            v22 = 3;
            v64.m128i_i64[0] = (__int64)"A pointer to ";
            v64.m128i_i64[1] = (__int64)"local";
            LOWORD(v65) = 771;
            goto LABEL_85;
          }
          if ( v21 != 72 )
          {
            v22 = 3;
            v64.m128i_i64[0] = (__int64)"A pointer to ";
            LOWORD(v65) = 259;
            goto LABEL_85;
          }
          if ( sub_1CCB220(v62) )
            break;
          v45 = sub_1CCB280(v62);
          v22 = v67;
          if ( v45 )
          {
            v23 = "shared";
            goto LABEL_57;
          }
          v64.m128i_i64[0] = (__int64)"A pointer to ";
          LOWORD(v65) = 259;
LABEL_58:
          if ( !v22 )
          {
            LOWORD(v70) = 256;
LABEL_60:
            LOWORD(v74) = 256;
LABEL_61:
            LOWORD(v78) = 256;
            goto LABEL_62;
          }
          if ( v22 == 1 )
          {
            v36 = v65;
            v69 = _mm_loadu_si128(&v64);
            v70 = v65;
            v38 = v72;
            if ( !(_BYTE)v72 )
              goto LABEL_60;
            goto LABEL_90;
          }
LABEL_85:
          v35 = &v64;
          v36 = 2;
          if ( BYTE1(v65) == 1 )
          {
            v35 = (__m128i *)v64.m128i_i64[0];
            v36 = 3;
          }
          v37 = (const char **)v66;
          if ( v68 != 1 )
          {
            v37 = &v66;
            v22 = 2;
          }
          BYTE1(v70) = v22;
          v38 = v72;
          v69.m128i_i64[0] = (__int64)v35;
          v69.m128i_i64[1] = (__int64)v37;
          LOBYTE(v70) = v36;
          if ( !(_BYTE)v72 )
            goto LABEL_60;
LABEL_90:
          if ( v38 == 1 )
          {
            v46 = _mm_loadu_si128(&v69);
            v74 = v70;
            v73 = v46;
            v36 = v70;
            if ( !(_BYTE)v70 )
              goto LABEL_61;
            v41 = v76;
            if ( !(_BYTE)v76 )
              goto LABEL_61;
            if ( (_BYTE)v74 == 1 )
            {
              v77 = _mm_loadu_si128(&v75);
              v78 = v76;
              goto LABEL_62;
            }
          }
          else
          {
            v39 = (__m128i *)v69.m128i_i64[0];
            if ( BYTE1(v70) != 1 )
            {
              v39 = &v69;
              v36 = 2;
            }
            v40 = (_QWORD *)v71[0];
            if ( BYTE1(v72) != 1 )
            {
              v40 = v71;
              v38 = 2;
            }
            BYTE1(v74) = v38;
            v41 = v76;
            v73.m128i_i64[0] = (__int64)v39;
            v73.m128i_i64[1] = (__int64)v40;
            LOBYTE(v74) = v36;
            if ( !(_BYTE)v76 )
              goto LABEL_61;
          }
          if ( v41 == 1 )
          {
            v77 = _mm_loadu_si128(&v73);
            v78 = v74;
          }
          else
          {
            v42 = (__m128i *)v73.m128i_i64[0];
            if ( BYTE1(v74) != 1 )
            {
              v42 = &v73;
              v36 = 2;
            }
            v43 = (__m128i *)v75.m128i_i64[0];
            if ( BYTE1(v76) != 1 )
            {
              v43 = &v75;
              v41 = 2;
            }
            v77.m128i_i64[0] = (__int64)v42;
            v77.m128i_i64[1] = (__int64)v43;
            LOBYTE(v78) = v36;
            BYTE1(v78) = v41;
          }
LABEL_62:
          sub_16E2FC0(v90, (__int64)&v77);
          v81 = 1;
          v85 = 260;
          v84 = v90;
          v79 = " : Warning: ";
          v80 = 3;
          sub_15E0530(*(_QWORD *)(v14[2] + 56LL));
          sub_1C315E0((__int64)v92, s2a);
          v24 = v80;
          if ( v80 )
          {
            if ( v80 == 1 )
            {
              LOWORD(v83) = 260;
              v82.m128i_i64[0] = (__int64)v92;
            }
            else
            {
              v25 = (const char **)v79;
              if ( v81 != 1 )
              {
                v25 = &v79;
                v24 = 2;
              }
              v82.m128i_i64[1] = (__int64)v25;
              LOBYTE(v83) = 4;
              v82.m128i_i64[0] = (__int64)v92;
              BYTE1(v83) = v24;
            }
            v26 = v85;
            if ( (_BYTE)v85 )
            {
              if ( (_BYTE)v85 == 1 )
              {
                v86 = _mm_loadu_si128(&v82);
                v87 = v83;
              }
              else
              {
                v27 = &v82;
                v28 = 2;
                if ( BYTE1(v83) == 1 )
                {
                  v27 = (__m128i *)v82.m128i_i64[0];
                  v28 = 4;
                }
                v29 = (__int64 **)v84;
                if ( HIBYTE(v85) != 1 )
                {
                  v29 = &v84;
                  v26 = 2;
                }
                v86.m128i_i64[0] = (__int64)v27;
                v86.m128i_i64[1] = (__int64)v29;
                LOBYTE(v87) = v28;
                BYTE1(v87) = v26;
              }
              goto LABEL_104;
            }
          }
          else
          {
            LOWORD(v83) = 256;
          }
          LOWORD(v87) = 256;
LABEL_104:
          sub_16E2FC0(v88, (__int64)&v86);
          if ( (_QWORD *)v92[0] != v93 )
            j_j___libc_free_0(v92[0], v93[0] + 1LL);
          if ( (__int64 *)v90[0] != &v91 )
            j_j___libc_free_0(v90[0], v91 + 1);
          sub_1C3F040((__int64)v88);
          if ( (__int64 *)v88[0] != &v89 )
            j_j___libc_free_0(v88[0], v89 + 1);
LABEL_46:
          if ( v13 == (_QWORD *)v3 )
            return;
          v14 = j;
        }
        v22 = v67;
        v23 = "local";
LABEL_57:
        v64.m128i_i64[1] = (__int64)v23;
        v64.m128i_i64[0] = (__int64)"A pointer to ";
        LOWORD(v65) = 771;
        goto LABEL_58;
      }
    }
  }
}
