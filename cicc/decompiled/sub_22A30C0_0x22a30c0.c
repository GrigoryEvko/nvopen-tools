// Function: sub_22A30C0
// Address: 0x22a30c0
//
void __fastcall sub_22A30C0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, char a5)
{
  char *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // rax
  void *v13; // rcx
  unsigned __int64 *v14; // rdx
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // r12
  char v17; // dl
  _BYTE *p_src; // rdi
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rax
  _DWORD *v24; // rdx
  char *v25; // rsi
  __int64 v26; // rax
  _BYTE *v27; // rdx
  const char *v28; // rax
  __int64 v29; // rdx
  char v30; // al
  __m128i *v31; // rcx
  _QWORD *v32; // rdx
  char v33; // al
  char v34; // dl
  _QWORD *v35; // rsi
  __int64 v36; // rdi
  _WORD *v37; // rdx
  _QWORD *v38; // rdi
  _BYTE *v39; // rax
  _QWORD *v40; // rax
  __m128i *v41; // rdx
  __m128i si128; // xmm0
  size_t v43; // rdx
  __m128i v44; // xmm5
  __int64 v45; // [rsp+0h] [rbp-240h]
  __int64 v46; // [rsp+8h] [rbp-238h]
  __int64 v47; // [rsp+10h] [rbp-230h]
  __int64 v50; // [rsp+38h] [rbp-208h] BYREF
  int v51; // [rsp+40h] [rbp-200h] BYREF
  __int64 (__fastcall **v52)(); // [rsp+48h] [rbp-1F8h]
  __int64 v53[2]; // [rsp+50h] [rbp-1F0h] BYREF
  char v54; // [rsp+60h] [rbp-1E0h]
  char v55; // [rsp+61h] [rbp-1DFh]
  void *dest; // [rsp+70h] [rbp-1D0h] BYREF
  unsigned __int64 v57; // [rsp+78h] [rbp-1C8h]
  __m128i v58; // [rsp+80h] [rbp-1C0h] BYREF
  _BYTE *v59; // [rsp+90h] [rbp-1B0h] BYREF
  size_t v60; // [rsp+98h] [rbp-1A8h]
  _QWORD v61[2]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v62[2]; // [rsp+B0h] [rbp-190h] BYREF
  _QWORD v63[2]; // [rsp+C0h] [rbp-180h] BYREF
  __int64 *v64; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v65; // [rsp+D8h] [rbp-168h]
  _QWORD v66[2]; // [rsp+E0h] [rbp-160h] BYREF
  const char *v67; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v68; // [rsp+F8h] [rbp-148h]
  __int16 v69; // [rsp+110h] [rbp-130h]
  __m128i v70; // [rsp+120h] [rbp-120h] BYREF
  __m128i v71; // [rsp+130h] [rbp-110h] BYREF
  __int64 v72; // [rsp+140h] [rbp-100h]
  _QWORD v73[4]; // [rsp+150h] [rbp-F0h] BYREF
  char v74; // [rsp+170h] [rbp-D0h]
  char v75; // [rsp+171h] [rbp-CFh]
  size_t n[2]; // [rsp+180h] [rbp-C0h] BYREF
  __m128i src; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+1A0h] [rbp-A0h]
  _BYTE *v79; // [rsp+1B0h] [rbp-90h] BYREF
  size_t v80; // [rsp+1B8h] [rbp-88h]
  _OWORD v81[8]; // [rsp+1C0h] [rbp-80h] BYREF

  v50 = a2;
  v7 = (char *)sub_BD5D20(a1);
  if ( v7 )
  {
    v70.m128i_i64[0] = (__int64)&v71;
    sub_229AB90(v70.m128i_i64, v7, (__int64)&v7[v8]);
  }
  else
  {
    v71.m128i_i8[0] = 0;
    v70 = (__m128i)(unsigned __int64)&v71;
  }
  if ( a3 )
  {
    v64 = v66;
    sub_229AB90((__int64 *)&v64, a3, (__int64)&a3[a4]);
    if ( v65 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_93;
  }
  else
  {
    LOBYTE(v66[0]) = 0;
    v64 = v66;
    v65 = 0;
  }
  v9 = sub_2241490((unsigned __int64 *)&v64, ".", 1u);
  v79 = v81;
  if ( (unsigned __int64 *)*v9 == v9 + 2 )
  {
    v81[0] = _mm_loadu_si128((const __m128i *)v9 + 1);
  }
  else
  {
    v79 = (_BYTE *)*v9;
    *(_QWORD *)&v81[0] = v9[2];
  }
  v80 = v9[1];
  *v9 = (unsigned __int64)(v9 + 2);
  v9[1] = 0;
  *((_BYTE *)v9 + 16) = 0;
  v10 = 15;
  v11 = 15;
  if ( v79 != (_BYTE *)v81 )
    v11 = *(_QWORD *)&v81[0];
  if ( v80 + v70.m128i_i64[1] <= v11 )
    goto LABEL_13;
  if ( (__m128i *)v70.m128i_i64[0] != &v71 )
    v10 = v71.m128i_i64[0];
  if ( v80 + v70.m128i_i64[1] <= v10 )
  {
    v12 = sub_2241130((unsigned __int64 *)&v70, 0, 0, v79, v80);
    v14 = v12 + 2;
    dest = &v58;
    v13 = (void *)*v12;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
      goto LABEL_14;
  }
  else
  {
LABEL_13:
    v12 = sub_2241490((unsigned __int64 *)&v79, (char *)v70.m128i_i64[0], v70.m128i_u64[1]);
    dest = &v58;
    v13 = (void *)*v12;
    v14 = v12 + 2;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
    {
LABEL_14:
      dest = v13;
      v58.m128i_i64[0] = v12[2];
      goto LABEL_15;
    }
  }
  v58 = _mm_loadu_si128((const __m128i *)v12 + 1);
LABEL_15:
  v57 = v12[1];
  *v12 = (unsigned __int64)v14;
  v12[1] = 0;
  *((_BYTE *)v12 + 16) = 0;
  if ( v79 != (_BYTE *)v81 )
    j_j___libc_free_0((unsigned __int64)v79);
  if ( v64 != v66 )
    j_j___libc_free_0((unsigned __int64)v64);
  if ( (__m128i *)v70.m128i_i64[0] != &v71 )
    j_j___libc_free_0(v70.m128i_u64[0]);
  v15 = v57;
  if ( v57 > 0xFA )
  {
    sub_22410F0((unsigned __int64 *)&dest, 0xFAu, 0);
    v15 = v57;
  }
  if ( v15 )
  {
    v16 = v15;
    do
    {
      sub_229AC40((unsigned __int64 *)&qword_4FDB640, (__int64)&dest, 1);
      if ( v17 )
        break;
      sub_22410F0((unsigned __int64 *)&dest, (unsigned __int8)(v16-- + -7 - v15), 0);
    }
    while ( v16 );
    v15 = v57;
  }
  n[0] = (size_t)&src;
  sub_229AAE0((__int64 *)n, dest, (__int64)dest + v15);
  if ( 0x3FFFFFFFFFFFFFFFLL - n[1] <= 3 )
    goto LABEL_93;
  sub_2241490(n, ".dot", 4u);
  p_src = dest;
  if ( (__m128i *)n[0] == &src )
  {
    v43 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *(_BYTE *)dest = src.m128i_i8[0];
      else
        memcpy(dest, &src, n[1]);
      v43 = n[1];
      p_src = dest;
    }
    v57 = v43;
    p_src[v43] = 0;
    p_src = (_BYTE *)n[0];
  }
  else
  {
    if ( dest == &v58 )
    {
      dest = (void *)n[0];
      v57 = n[1];
      v58.m128i_i64[0] = src.m128i_i64[0];
    }
    else
    {
      v19 = v58.m128i_i64[0];
      dest = (void *)n[0];
      v57 = n[1];
      v58.m128i_i64[0] = src.m128i_i64[0];
      if ( p_src )
      {
        n[0] = (size_t)p_src;
        src.m128i_i64[0] = v19;
        goto LABEL_33;
      }
    }
    n[0] = (size_t)&src;
    p_src = &src;
  }
LABEL_33:
  n[1] = 0;
  *p_src = 0;
  if ( (__m128i *)n[0] != &src )
    j_j___libc_free_0(n[0]);
  v51 = 0;
  v52 = sub_2241E40();
  v20 = sub_CB72A0();
  v21 = v20[4];
  v22 = (__int64)v20;
  if ( (unsigned __int64)(v20[3] - v21) <= 8 )
  {
    v22 = sub_CB6200((__int64)v20, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v21 + 8) = 39;
    *(_QWORD *)v21 = 0x20676E6974697257LL;
    v20[4] += 9LL;
  }
  v23 = sub_CB6200(v22, (unsigned __int8 *)dest, v57);
  v24 = *(_DWORD **)(v23 + 32);
  if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 3u )
  {
    sub_CB6200(v23, "'...", 4u);
  }
  else
  {
    *v24 = 774778407;
    *(_QWORD *)(v23 + 32) += 4LL;
  }
  sub_CB7060((__int64)&v79, dest, v57, (__int64)&v51, 3u);
  v25 = (char *)n;
  v59 = v61;
  n[0] = 19;
  v26 = sub_22409D0((__int64)&v59, n, 0);
  v59 = (_BYTE *)v26;
  v61[0] = n[0];
  *(__m128i *)v26 = _mm_load_si128((const __m128i *)&xmmword_4289C10);
  v27 = v59;
  *(_WORD *)(v26 + 16) = 25970;
  *(_BYTE *)(v26 + 18) = 101;
  v60 = n[0];
  v27[n[0]] = 0;
  if ( v51 )
  {
    v40 = sub_CB72A0();
    v41 = (__m128i *)v40[4];
    if ( v40[3] - (_QWORD)v41 <= 0x20u )
    {
      v25 = "  error opening file for writing!";
      sub_CB6200((__int64)v40, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v41[2].m128i_i8[0] = 33;
      *v41 = si128;
      v41[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v40[4] += 33LL;
    }
    goto LABEL_63;
  }
  v75 = 1;
  v73[0] = "' function";
  v74 = 3;
  v28 = sub_BD5D20(a1);
  v68 = v29;
  v67 = v28;
  v69 = 261;
  v62[0] = (__int64)v63;
  sub_229AAE0(v62, v59, (__int64)&v59[v60]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v62[1]) <= 5 )
LABEL_93:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v62, " for '", 6u);
  v30 = v69;
  v31 = (__m128i *)v62;
  if ( (_BYTE)v69 )
  {
    if ( (_BYTE)v69 == 1 )
    {
      v33 = v74;
      v70.m128i_i64[0] = (__int64)v62;
      LOWORD(v72) = 260;
      if ( v74 )
      {
        if ( v74 != 1 )
        {
          v34 = 4;
          v47 = v70.m128i_i64[1];
          goto LABEL_48;
        }
        goto LABEL_86;
      }
    }
    else
    {
      if ( HIBYTE(v69) == 1 )
      {
        v32 = v67;
        v46 = v68;
      }
      else
      {
        v32 = &v67;
        v30 = 2;
      }
      v70.m128i_i64[0] = (__int64)v62;
      BYTE1(v72) = v30;
      v33 = v74;
      v71.m128i_i64[0] = (__int64)v32;
      v71.m128i_i64[1] = v46;
      LOBYTE(v72) = 4;
      if ( v74 )
      {
        if ( v74 != 1 )
        {
          v31 = &v70;
          v34 = 2;
LABEL_48:
          if ( v75 == 1 )
          {
            v35 = (_QWORD *)v73[0];
            v45 = v73[1];
          }
          else
          {
            v35 = v73;
            v33 = 2;
          }
          n[0] = (size_t)v31;
          src.m128i_i64[0] = (__int64)v35;
          n[1] = v47;
          LOBYTE(v78) = v34;
          src.m128i_i64[1] = v45;
          BYTE1(v78) = v33;
          goto LABEL_57;
        }
LABEL_86:
        v44 = _mm_load_si128(&v71);
        *(__m128i *)n = _mm_load_si128(&v70);
        v78 = v72;
        src = v44;
        goto LABEL_57;
      }
    }
  }
  else
  {
    LOWORD(v72) = 256;
  }
  LOWORD(v78) = 256;
LABEL_57:
  v53[0] = (__int64)&v79;
  v53[1] = (__int64)&v50;
  v54 = 0;
  v55 = a5;
  sub_CA0F50((__int64 *)&v64, (void **)n);
  v25 = (char *)&v64;
  sub_229D570(v53, &v64);
  sub_229EF00((__int64)v53);
  v36 = v53[0];
  v37 = *(_WORD **)(v53[0] + 32);
  if ( *(_QWORD *)(v53[0] + 24) - (_QWORD)v37 <= 1u )
  {
    v25 = "}\n";
    sub_CB6200(v53[0], "}\n", 2u);
  }
  else
  {
    *v37 = 2685;
    *(_QWORD *)(v36 + 32) += 2LL;
  }
  if ( v64 != v66 )
  {
    v25 = (char *)(v66[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v64);
  }
  if ( (_QWORD *)v62[0] != v63 )
  {
    v25 = (char *)(v63[0] + 1LL);
    j_j___libc_free_0(v62[0]);
  }
LABEL_63:
  v38 = sub_CB72A0();
  v39 = (_BYTE *)v38[4];
  if ( (_BYTE *)v38[3] == v39 )
  {
    v25 = "\n";
    sub_CB6200((__int64)v38, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v39 = 10;
    ++v38[4];
  }
  if ( v59 != (_BYTE *)v61 )
  {
    v25 = (char *)(v61[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v59);
  }
  sub_CB5B00((int *)&v79, (__int64)v25);
  if ( dest != &v58 )
    j_j___libc_free_0((unsigned __int64)dest);
}
