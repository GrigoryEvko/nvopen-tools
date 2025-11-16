// Function: sub_155D8D0
// Address: 0x155d8d0
//
__int64 __fastcall sub_155D8D0(__int64 a1, __int64 *a2, char a3)
{
  bool v4; // zf
  __m128i *v6; // rax
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __m128i si128; // xmm0
  __int64 v15; // rax
  __int64 v16; // rdx
  const char *v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // ebx
  unsigned int v23; // r14d
  __int64 v24; // rcx
  unsigned __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int8 v29; // al
  __int64 v30; // rcx
  __int8 *v31; // rax
  unsigned __int64 v32; // rdx
  __int8 *v33; // r9
  size_t v34; // rbx
  __m128i *v35; // rdi
  __int64 v36; // rbx
  __m128i *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int8 *v41; // [rsp+8h] [rbp-138h]
  char v42; // [rsp+1Ch] [rbp-124h] BYREF
  _QWORD v43[2]; // [rsp+20h] [rbp-120h] BYREF
  _QWORD v44[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v45; // [rsp+40h] [rbp-100h]
  __m128i v46; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+70h] [rbp-D0h]
  __int64 v48; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v49; // [rsp+90h] [rbp-B0h]
  __m128i *v50; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-98h]
  __m128i v52; // [rsp+B0h] [rbp-90h] BYREF
  __m128i p_dest; // [rsp+C0h] [rbp-80h] BYREF
  __m128i dest; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int64 v55; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v56; // [rsp+E8h] [rbp-58h]
  __int64 v57; // [rsp+F0h] [rbp-50h]
  __int64 v58; // [rsp+F8h] [rbp-48h]
  int v59; // [rsp+100h] [rbp-40h]
  __m128i **v60; // [rsp+108h] [rbp-38h]

  v4 = *a2 == 0;
  v42 = a3;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( !sub_155D460(a2, 42) )
  {
    if ( sub_155D460(a2, 43) )
    {
      *(_QWORD *)a1 = a1 + 16;
      v55 = 18;
      v12 = sub_22409D0(a1, &v55, 0);
      v13 = v55;
      si128 = _mm_load_si128((const __m128i *)&xmmword_4293170);
      *(_QWORD *)a1 = v12;
      *(_QWORD *)(a1 + 16) = v13;
      *(_WORD *)(v12 + 16) = 29555;
      *(__m128i *)v12 = si128;
      v15 = v55;
      v16 = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 8) = v55;
      *(_BYTE *)(v16 + v15) = 0;
      return a1;
    }
    if ( sub_155D460(a2, 3) )
    {
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "alwaysinline");
      *(_QWORD *)(a1 + 8) = 12;
      return a1;
    }
    if ( sub_155D460(a2, 4) )
    {
      *(_QWORD *)(a1 + 8) = 10;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "argmemonly");
      return a1;
    }
    if ( sub_155D460(a2, 5) )
    {
      *(_DWORD *)(a1 + 16) = 1818850658;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 20) = 26996;
      *(_BYTE *)(a1 + 22) = 110;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return a1;
    }
    if ( sub_155D460(a2, 6) )
    {
      *(_BYTE *)(a1 + 20) = 108;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1635154274;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    }
    if ( sub_155D460(a2, 8) )
    {
      *(_QWORD *)(a1 + 8) = 10;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "convergent");
      return a1;
    }
    if ( sub_155D460(a2, 54) )
    {
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "swifterror");
      *(_QWORD *)(a1 + 8) = 10;
      return a1;
    }
    if ( sub_155D460(a2, 55) )
    {
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "swiftself");
      *(_QWORD *)(a1 + 8) = 9;
      return a1;
    }
    if ( sub_155D460(a2, 13) )
    {
      sub_155CC90((__int64 *)a1, "inaccessiblememonly");
      return a1;
    }
    if ( sub_155D460(a2, 14) )
    {
      sub_155CC90((__int64 *)a1, "inaccessiblemem_or_argmemonly");
      return a1;
    }
    if ( sub_155D460(a2, 11) )
    {
      sub_155CC90((__int64 *)a1, "inalloca");
      return a1;
    }
    if ( sub_155D460(a2, 15) )
    {
      sub_155CC90((__int64 *)a1, "inlinehint");
      return a1;
    }
    if ( sub_155D460(a2, 12) )
    {
      sub_155CC90((__int64 *)a1, "inreg");
      return a1;
    }
    if ( sub_155D460(a2, 16) )
    {
      sub_155CC90((__int64 *)a1, "jumptable");
      return a1;
    }
    if ( sub_155D460(a2, 17) )
    {
      sub_155CC90((__int64 *)a1, "minsize");
      return a1;
    }
    if ( sub_155D460(a2, 18) )
    {
      sub_155CC90((__int64 *)a1, "naked");
      return a1;
    }
    if ( sub_155D460(a2, 19) )
    {
      sub_155CC90((__int64 *)a1, "nest");
      return a1;
    }
    if ( sub_155D460(a2, 20) )
    {
      sub_155CC90((__int64 *)a1, "noalias");
      return a1;
    }
    if ( sub_155D460(a2, 21) )
    {
      sub_155CC90((__int64 *)a1, "nobuiltin");
      return a1;
    }
    if ( sub_155D460(a2, 22) )
    {
      sub_155CC90((__int64 *)a1, "nocapture");
      return a1;
    }
    if ( sub_155D460(a2, 24) )
    {
      sub_155CC90((__int64 *)a1, "noduplicate");
      return a1;
    }
    if ( sub_155D460(a2, 25) )
    {
      sub_155CC90((__int64 *)a1, "noimplicitfloat");
      return a1;
    }
    if ( sub_155D460(a2, 26) )
    {
      sub_155CC90((__int64 *)a1, "noinline");
      return a1;
    }
    if ( sub_155D460(a2, 31) )
    {
      sub_155CC90((__int64 *)a1, "nonlazybind");
      return a1;
    }
    if ( sub_155D460(a2, 32) )
    {
      sub_155CC90((__int64 *)a1, "nonnull");
      return a1;
    }
    if ( sub_155D460(a2, 28) )
    {
      sub_155CC90((__int64 *)a1, "noredzone");
      return a1;
    }
    if ( sub_155D460(a2, 29) )
    {
      sub_155CC90((__int64 *)a1, "noreturn");
      return a1;
    }
    if ( sub_155D460(a2, 23) )
    {
      sub_155CC90((__int64 *)a1, "nocf_check");
      return a1;
    }
    if ( sub_155D460(a2, 27) )
    {
      sub_155CC90((__int64 *)a1, "norecurse");
      return a1;
    }
    if ( sub_155D460(a2, 30) )
    {
      sub_155CC90((__int64 *)a1, "nounwind");
      return a1;
    }
    if ( sub_155D460(a2, 33) )
    {
      sub_155CC90((__int64 *)a1, "optforfuzzing");
      return a1;
    }
    if ( sub_155D460(a2, 35) )
    {
      sub_155CC90((__int64 *)a1, "optnone");
      return a1;
    }
    if ( sub_155D460(a2, 34) )
    {
      sub_155CC90((__int64 *)a1, "optsize");
      return a1;
    }
    if ( sub_155D460(a2, 36) )
    {
      sub_155CC90((__int64 *)a1, "readnone");
      return a1;
    }
    if ( sub_155D460(a2, 37) )
    {
      sub_155CC90((__int64 *)a1, "readonly");
      return a1;
    }
    if ( sub_155D460(a2, 57) )
    {
      sub_155CC90((__int64 *)a1, "writeonly");
      return a1;
    }
    if ( sub_155D460(a2, 38) )
    {
      sub_155CC90((__int64 *)a1, "returned");
      return a1;
    }
    if ( sub_155D460(a2, 39) )
    {
      sub_155CC90((__int64 *)a1, "returns_twice");
      return a1;
    }
    if ( sub_155D460(a2, 40) )
    {
      sub_155CC90((__int64 *)a1, "signext");
      return a1;
    }
    if ( sub_155D460(a2, 47) )
    {
      sub_155CC90((__int64 *)a1, "speculatable");
      return a1;
    }
    if ( sub_155D460(a2, 49) )
    {
      sub_155CC90((__int64 *)a1, "ssp");
      return a1;
    }
    if ( sub_155D460(a2, 50) )
    {
      sub_155CC90((__int64 *)a1, "sspreq");
      return a1;
    }
    if ( sub_155D460(a2, 51) )
    {
      sub_155CC90((__int64 *)a1, "sspstrong");
      return a1;
    }
    if ( sub_155D460(a2, 41) )
    {
      sub_155CC90((__int64 *)a1, "safestack");
      return a1;
    }
    if ( sub_155D460(a2, 46) )
    {
      sub_155CC90((__int64 *)a1, "shadowcallstack");
      return a1;
    }
    if ( sub_155D460(a2, 52) )
    {
      sub_155CC90((__int64 *)a1, "strictfp");
      return a1;
    }
    if ( sub_155D460(a2, 53) )
    {
      sub_155CC90((__int64 *)a1, "sret");
      return a1;
    }
    if ( sub_155D460(a2, 45) )
    {
      sub_155CC90((__int64 *)a1, "sanitize_thread");
      return a1;
    }
    if ( sub_155D460(a2, 44) )
    {
      sub_155CC90((__int64 *)a1, "sanitize_memory");
      return a1;
    }
    if ( sub_155D460(a2, 56) )
    {
      sub_155CC90((__int64 *)a1, "uwtable");
      return a1;
    }
    if ( sub_155D460(a2, 58) )
    {
      sub_155CC90((__int64 *)a1, "zeroext");
      return a1;
    }
    if ( sub_155D460(a2, 7) )
    {
      sub_155CC90((__int64 *)a1, "cold");
      return a1;
    }
    if ( sub_155D460(a2, 1) )
    {
      dest.m128i_i8[0] = 0;
      p_dest = (__m128i)(unsigned __int64)&dest;
      sub_155CA90((__int64)&p_dest, "align");
      v17 = "=";
      if ( !v42 )
        v17 = " ";
      sub_155CA90((__int64)&p_dest, v17);
      v18 = sub_155D4B0(a2);
      sub_155CE20((__int64 *)&v55, v18, 0);
      sub_2241490(&p_dest, v55, v56, v19);
      sub_2240A30(&v55);
      *(_QWORD *)a1 = a1 + 16;
      v20 = p_dest.m128i_i64[0];
      if ( (__m128i *)p_dest.m128i_i64[0] != &dest )
        goto LABEL_114;
      goto LABEL_127;
    }
    v43[1] = a2;
    v43[0] = &v42;
    if ( sub_155D460(a2, 48) )
    {
      sub_155D4D0(a1, (__int64)v43, "alignstack");
      return a1;
    }
    if ( sub_155D460(a2, 9) )
    {
      sub_155D4D0(a1, (__int64)v43, "dereferenceable");
      return a1;
    }
    if ( sub_155D460(a2, 10) )
    {
      sub_155D4D0(a1, (__int64)v43, "dereferenceable_or_null");
      return a1;
    }
    if ( sub_155D460(a2, 2) )
    {
      sub_155D750((__int64)&v55, a2);
      v22 = v55;
      if ( (_BYTE)v56 )
      {
        v23 = HIDWORD(v55);
        sub_155CC90(p_dest.m128i_i64, "allocsize(");
        sub_155CE20((__int64 *)&v55, v22, 0);
        sub_2241490(&p_dest, v55, v56, v24);
        sub_2240A30(&v55);
        sub_2240F50(&p_dest, 44);
        v25 = v23;
      }
      else
      {
        sub_155CC90(p_dest.m128i_i64, "allocsize(");
        v25 = v22;
      }
      sub_155CE20((__int64 *)&v55, v25, 0);
      sub_2241490(&p_dest, v55, v56, v26);
      sub_2240A30(&v55);
      sub_2240F50(&p_dest, 41);
      *(_QWORD *)a1 = a1 + 16;
      v20 = p_dest.m128i_i64[0];
      if ( (__m128i *)p_dest.m128i_i64[0] != &dest )
      {
LABEL_114:
        *(_QWORD *)a1 = v20;
        *(_QWORD *)(a1 + 16) = dest.m128i_i64[0];
LABEL_115:
        v21 = p_dest.m128i_i64[1];
        p_dest = (__m128i)(unsigned __int64)&dest;
        *(_QWORD *)(a1 + 8) = v21;
        dest.m128i_i8[0] = 0;
        sub_2240A30(&p_dest);
        return a1;
      }
LABEL_127:
      *(__m128i *)(a1 + 16) = _mm_load_si128(&dest);
      goto LABEL_115;
    }
    sub_155D3E0((__int64)a2);
    v50 = &v52;
    v51 = 0;
    v52.m128i_i8[0] = 0;
    v48 = 34;
    v49 = 264;
    v27 = sub_155D7D0(a2);
    LOBYTE(v45) = 34;
    v44[0] = v27;
    v44[1] = v28;
    v46.m128i_i64[0] = v45;
    v46.m128i_i64[1] = (__int64)v44;
    v29 = v49;
    LOWORD(v47) = 1288;
    if ( (_BYTE)v49 )
    {
      if ( (_BYTE)v49 == 1 )
      {
        p_dest = _mm_load_si128(&v46);
        dest.m128i_i64[0] = v47;
      }
      else
      {
        if ( HIBYTE(v49) == 1 )
        {
          v40 = (__int64 *)v48;
        }
        else
        {
          v40 = &v48;
          v29 = 2;
        }
        p_dest.m128i_i64[1] = (__int64)v40;
        p_dest.m128i_i64[0] = (__int64)&v46;
        dest.m128i_i8[0] = 2;
        dest.m128i_i8[1] = v29;
      }
    }
    else
    {
      dest.m128i_i16[0] = 256;
    }
    sub_16E2FC0(&v55, &p_dest);
    sub_2241490(&v50, v55, v56, v30);
    sub_2240A30(&v55);
    v31 = (__int8 *)sub_155D8A0(*a2);
    v33 = v31;
    v34 = v32;
    if ( !v31 )
    {
      p_dest = (__m128i)(unsigned __int64)&dest;
      dest.m128i_i8[0] = 0;
LABEL_137:
      v36 = a1 + 16;
      if ( p_dest.m128i_i64[1] )
      {
        v59 = 1;
        v58 = 0;
        v57 = 0;
        v55 = (unsigned __int64)&unk_49EFBE0;
        v56 = 0;
        v60 = &v50;
        sub_155CAE0((__int64)&v55, "=\"");
        sub_16D16F0(p_dest.m128i_i64[0], p_dest.m128i_i64[1], &v55);
        sub_155CAE0((__int64)&v55, "\"");
        sub_16E7BC0(&v55);
        v37 = v50;
        *(_QWORD *)a1 = v36;
        if ( v37 != &v52 )
          goto LABEL_139;
      }
      else
      {
        v37 = v50;
        *(_QWORD *)a1 = v36;
        if ( v37 != &v52 )
        {
LABEL_139:
          *(_QWORD *)a1 = v37;
          *(_QWORD *)(a1 + 16) = v52.m128i_i64[0];
LABEL_140:
          v38 = v51;
          v50 = &v52;
          v51 = 0;
          *(_QWORD *)(a1 + 8) = v38;
          v52.m128i_i8[0] = 0;
          sub_2240A30(&p_dest);
          sub_2240A30(&v50);
          return a1;
        }
      }
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v52);
      goto LABEL_140;
    }
    v55 = v32;
    p_dest.m128i_i64[0] = (__int64)&dest;
    if ( v32 > 0xF )
    {
      v41 = v31;
      v39 = sub_22409D0(&p_dest, &v55, 0);
      v33 = v41;
      p_dest.m128i_i64[0] = v39;
      v35 = (__m128i *)v39;
      dest.m128i_i64[0] = v55;
    }
    else
    {
      if ( v32 == 1 )
      {
        dest.m128i_i8[0] = *v31;
        goto LABEL_136;
      }
      if ( !v32 )
      {
LABEL_136:
        p_dest.m128i_i64[1] = v55;
        *(_BYTE *)(p_dest.m128i_i64[0] + v55) = 0;
        goto LABEL_137;
      }
      v35 = &dest;
    }
    memcpy(v35, v33, v34);
    goto LABEL_136;
  }
  *(_QWORD *)a1 = a1 + 16;
  v55 = 16;
  v6 = (__m128i *)sub_22409D0(a1, &v55, 0);
  v7 = v55;
  v8 = _mm_load_si128((const __m128i *)&xmmword_4293160);
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 16) = v7;
  *v6 = v8;
  v9 = v55;
  v10 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 8) = v55;
  *(_BYTE *)(v10 + v9) = 0;
  return a1;
}
