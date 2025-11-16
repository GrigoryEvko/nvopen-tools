// Function: sub_22A2570
// Address: 0x22a2570
//
void __fastcall sub_22A2570(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, char a5)
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
  const char *v26; // rax
  __int64 v27; // rdx
  char v28; // al
  __m128i *v29; // rcx
  _QWORD *v30; // rdx
  char v31; // al
  char v32; // dl
  _QWORD *v33; // rsi
  __int64 v34; // rdi
  _WORD *v35; // rdx
  _QWORD *v36; // rdi
  _BYTE *v37; // rax
  _QWORD *v38; // rax
  __m128i *v39; // rdx
  __m128i si128; // xmm0
  size_t v41; // rdx
  __m128i v42; // xmm5
  __int64 v43; // [rsp+0h] [rbp-240h]
  __int64 v44; // [rsp+8h] [rbp-238h]
  __int64 v45; // [rsp+10h] [rbp-230h]
  __int64 v48; // [rsp+38h] [rbp-208h] BYREF
  int v49; // [rsp+40h] [rbp-200h] BYREF
  __int64 (__fastcall **v50)(); // [rsp+48h] [rbp-1F8h]
  __int64 v51[2]; // [rsp+50h] [rbp-1F0h] BYREF
  char v52; // [rsp+60h] [rbp-1E0h]
  char v53; // [rsp+61h] [rbp-1DFh]
  void *dest; // [rsp+70h] [rbp-1D0h] BYREF
  unsigned __int64 v55; // [rsp+78h] [rbp-1C8h]
  __m128i v56; // [rsp+80h] [rbp-1C0h] BYREF
  _BYTE *v57; // [rsp+90h] [rbp-1B0h]
  __int64 v58; // [rsp+98h] [rbp-1A8h]
  _QWORD v59[2]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v60[2]; // [rsp+B0h] [rbp-190h] BYREF
  _QWORD v61[2]; // [rsp+C0h] [rbp-180h] BYREF
  char *v62; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v63; // [rsp+D8h] [rbp-168h]
  _QWORD v64[2]; // [rsp+E0h] [rbp-160h] BYREF
  const char *v65; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v66; // [rsp+F8h] [rbp-148h]
  __int16 v67; // [rsp+110h] [rbp-130h]
  __m128i v68; // [rsp+120h] [rbp-120h] BYREF
  __m128i v69; // [rsp+130h] [rbp-110h] BYREF
  __int64 v70; // [rsp+140h] [rbp-100h]
  _QWORD v71[4]; // [rsp+150h] [rbp-F0h] BYREF
  char v72; // [rsp+170h] [rbp-D0h]
  char v73; // [rsp+171h] [rbp-CFh]
  size_t n[2]; // [rsp+180h] [rbp-C0h] BYREF
  __m128i src; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+1A0h] [rbp-A0h]
  _BYTE *v77; // [rsp+1B0h] [rbp-90h] BYREF
  size_t v78; // [rsp+1B8h] [rbp-88h]
  _OWORD v79[8]; // [rsp+1C0h] [rbp-80h] BYREF

  v48 = a2;
  v7 = (char *)sub_BD5D20(a1);
  if ( v7 )
  {
    v68.m128i_i64[0] = (__int64)&v69;
    sub_229AB90(v68.m128i_i64, v7, (__int64)&v7[v8]);
  }
  else
  {
    v69.m128i_i8[0] = 0;
    v68 = (__m128i)(unsigned __int64)&v69;
  }
  if ( a3 )
  {
    v62 = (char *)v64;
    sub_229AB90((__int64 *)&v62, a3, (__int64)&a3[a4]);
    if ( v63 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_93;
  }
  else
  {
    LOBYTE(v64[0]) = 0;
    v62 = (char *)v64;
    v63 = 0;
  }
  v9 = sub_2241490((unsigned __int64 *)&v62, ".", 1u);
  v77 = v79;
  if ( (unsigned __int64 *)*v9 == v9 + 2 )
  {
    v79[0] = _mm_loadu_si128((const __m128i *)v9 + 1);
  }
  else
  {
    v77 = (_BYTE *)*v9;
    *(_QWORD *)&v79[0] = v9[2];
  }
  v78 = v9[1];
  *v9 = (unsigned __int64)(v9 + 2);
  v9[1] = 0;
  *((_BYTE *)v9 + 16) = 0;
  v10 = 15;
  v11 = 15;
  if ( v77 != (_BYTE *)v79 )
    v11 = *(_QWORD *)&v79[0];
  if ( v78 + v68.m128i_i64[1] <= v11 )
    goto LABEL_13;
  if ( (__m128i *)v68.m128i_i64[0] != &v69 )
    v10 = v69.m128i_i64[0];
  if ( v78 + v68.m128i_i64[1] <= v10 )
  {
    v12 = sub_2241130((unsigned __int64 *)&v68, 0, 0, v77, v78);
    v14 = v12 + 2;
    dest = &v56;
    v13 = (void *)*v12;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
      goto LABEL_14;
  }
  else
  {
LABEL_13:
    v12 = sub_2241490((unsigned __int64 *)&v77, (char *)v68.m128i_i64[0], v68.m128i_u64[1]);
    dest = &v56;
    v13 = (void *)*v12;
    v14 = v12 + 2;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
    {
LABEL_14:
      dest = v13;
      v56.m128i_i64[0] = v12[2];
      goto LABEL_15;
    }
  }
  v56 = _mm_loadu_si128((const __m128i *)v12 + 1);
LABEL_15:
  v55 = v12[1];
  *v12 = (unsigned __int64)v14;
  v12[1] = 0;
  *((_BYTE *)v12 + 16) = 0;
  if ( v77 != (_BYTE *)v79 )
    j_j___libc_free_0((unsigned __int64)v77);
  if ( v62 != (char *)v64 )
    j_j___libc_free_0((unsigned __int64)v62);
  if ( (__m128i *)v68.m128i_i64[0] != &v69 )
    j_j___libc_free_0(v68.m128i_u64[0]);
  v15 = v55;
  if ( v55 > 0xFA )
  {
    sub_22410F0((unsigned __int64 *)&dest, 0xFAu, 0);
    v15 = v55;
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
    v15 = v55;
  }
  n[0] = (size_t)&src;
  sub_229AAE0((__int64 *)n, dest, (__int64)dest + v15);
  if ( 0x3FFFFFFFFFFFFFFFLL - n[1] <= 3 )
    goto LABEL_93;
  sub_2241490(n, ".dot", 4u);
  p_src = dest;
  if ( (__m128i *)n[0] == &src )
  {
    v41 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *(_BYTE *)dest = src.m128i_i8[0];
      else
        memcpy(dest, &src, n[1]);
      v41 = n[1];
      p_src = dest;
    }
    v55 = v41;
    p_src[v41] = 0;
    p_src = (_BYTE *)n[0];
  }
  else
  {
    if ( dest == &v56 )
    {
      dest = (void *)n[0];
      v55 = n[1];
      v56.m128i_i64[0] = src.m128i_i64[0];
    }
    else
    {
      v19 = v56.m128i_i64[0];
      dest = (void *)n[0];
      v55 = n[1];
      v56.m128i_i64[0] = src.m128i_i64[0];
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
  v49 = 0;
  v50 = sub_2241E40();
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
  v23 = sub_CB6200(v22, (unsigned __int8 *)dest, v55);
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
  v25 = (char *)dest;
  sub_CB7060((__int64)&v77, dest, v55, (__int64)&v49, 3u);
  v57 = v59;
  strcpy((char *)v59, "Dominator tree");
  v58 = 14;
  if ( v49 )
  {
    v38 = sub_CB72A0();
    v39 = (__m128i *)v38[4];
    if ( v38[3] - (_QWORD)v39 <= 0x20u )
    {
      v25 = "  error opening file for writing!";
      sub_CB6200((__int64)v38, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v39[2].m128i_i8[0] = 33;
      *v39 = si128;
      v39[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v38[4] += 33LL;
    }
    goto LABEL_63;
  }
  v73 = 1;
  v71[0] = "' function";
  v72 = 3;
  v26 = sub_BD5D20(a1);
  v66 = v27;
  v65 = v26;
  v67 = 261;
  v60[0] = (__int64)v61;
  sub_229AAE0(v60, v57, (__int64)&v57[v58]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v60[1]) <= 5 )
LABEL_93:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v60, " for '", 6u);
  v28 = v67;
  v29 = (__m128i *)v60;
  if ( (_BYTE)v67 )
  {
    if ( (_BYTE)v67 == 1 )
    {
      v31 = v72;
      v68.m128i_i64[0] = (__int64)v60;
      LOWORD(v70) = 260;
      if ( v72 )
      {
        if ( v72 != 1 )
        {
          v32 = 4;
          v45 = v68.m128i_i64[1];
          goto LABEL_48;
        }
        goto LABEL_86;
      }
    }
    else
    {
      if ( HIBYTE(v67) == 1 )
      {
        v30 = v65;
        v44 = v66;
      }
      else
      {
        v30 = &v65;
        v28 = 2;
      }
      v68.m128i_i64[0] = (__int64)v60;
      BYTE1(v70) = v28;
      v31 = v72;
      v69.m128i_i64[0] = (__int64)v30;
      v69.m128i_i64[1] = v44;
      LOBYTE(v70) = 4;
      if ( v72 )
      {
        if ( v72 != 1 )
        {
          v29 = &v68;
          v32 = 2;
LABEL_48:
          if ( v73 == 1 )
          {
            v33 = (_QWORD *)v71[0];
            v43 = v71[1];
          }
          else
          {
            v33 = v71;
            v31 = 2;
          }
          n[0] = (size_t)v29;
          src.m128i_i64[0] = (__int64)v33;
          n[1] = v45;
          LOBYTE(v76) = v32;
          src.m128i_i64[1] = v43;
          BYTE1(v76) = v31;
          goto LABEL_57;
        }
LABEL_86:
        v42 = _mm_load_si128(&v69);
        *(__m128i *)n = _mm_load_si128(&v68);
        v76 = v70;
        src = v42;
        goto LABEL_57;
      }
    }
  }
  else
  {
    LOWORD(v70) = 256;
  }
  LOWORD(v76) = 256;
LABEL_57:
  v51[0] = (__int64)&v77;
  v51[1] = (__int64)&v48;
  v52 = 0;
  v53 = a5;
  sub_CA0F50((__int64 *)&v62, (void **)n);
  v25 = (char *)&v62;
  sub_229D920(v51, &v62);
  sub_22A07D0((__int64)v51);
  v34 = v51[0];
  v35 = *(_WORD **)(v51[0] + 32);
  if ( *(_QWORD *)(v51[0] + 24) - (_QWORD)v35 <= 1u )
  {
    v25 = "}\n";
    sub_CB6200(v51[0], "}\n", 2u);
  }
  else
  {
    *v35 = 2685;
    *(_QWORD *)(v34 + 32) += 2LL;
  }
  if ( v62 != (char *)v64 )
  {
    v25 = (char *)(v64[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v62);
  }
  if ( (_QWORD *)v60[0] != v61 )
  {
    v25 = (char *)(v61[0] + 1LL);
    j_j___libc_free_0(v60[0]);
  }
LABEL_63:
  v36 = sub_CB72A0();
  v37 = (_BYTE *)v36[4];
  if ( (_BYTE *)v36[3] == v37 )
  {
    v25 = "\n";
    sub_CB6200((__int64)v36, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v37 = 10;
    ++v36[4];
  }
  if ( v57 != (_BYTE *)v59 )
  {
    v25 = (char *)(v59[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v57);
  }
  sub_CB5B00((int *)&v77, (__int64)v25);
  if ( dest != &v56 )
    j_j___libc_free_0((unsigned __int64)dest);
}
