// Function: sub_311D9B0
// Address: 0x311d9b0
//
__int64 __fastcall sub_311D9B0(__int64 *a1, unsigned __int64 **a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // rax
  size_t v6; // rcx
  size_t v7; // rcx
  __int64 v8; // rsi
  unsigned __int64 *v9; // rax
  size_t v10; // rcx
  unsigned int v11; // r12d
  size_t v12; // r13
  unsigned __int64 v13; // rax
  __m128i *v14; // r14
  unsigned __int64 v15; // rax
  char v17; // r13
  size_t v18; // rbx
  size_t v19; // r14
  size_t v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // rbx
  __m128i *v23; // r15
  void *v24; // rcx
  void *v25; // r8
  size_t v26; // rdx
  int v27; // eax
  signed __int64 v28; // rax
  signed __int64 v29; // rax
  void *v30; // [rsp+8h] [rbp-1C8h]
  size_t v31; // [rsp+10h] [rbp-1C0h]
  void *v32; // [rsp+18h] [rbp-1B8h]
  void *v33; // [rsp+18h] [rbp-1B8h]
  __m128i *v34; // [rsp+20h] [rbp-1B0h] BYREF
  size_t v35; // [rsp+28h] [rbp-1A8h]
  __m128i v36; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned __int8 v37; // [rsp+40h] [rbp-190h]
  __m128i *v38; // [rsp+50h] [rbp-180h] BYREF
  size_t v39; // [rsp+58h] [rbp-178h]
  __m128i v40; // [rsp+60h] [rbp-170h] BYREF
  char v41; // [rsp+70h] [rbp-160h]
  __m128i *v42; // [rsp+80h] [rbp-150h] BYREF
  size_t v43; // [rsp+88h] [rbp-148h]
  __m128i v44; // [rsp+90h] [rbp-140h] BYREF
  char v45; // [rsp+A0h] [rbp-130h]
  __m128i *v46; // [rsp+B0h] [rbp-120h] BYREF
  size_t v47; // [rsp+B8h] [rbp-118h]
  __m128i v48; // [rsp+C0h] [rbp-110h] BYREF
  char v49; // [rsp+D0h] [rbp-100h]
  void *s1; // [rsp+E0h] [rbp-F0h]
  size_t n; // [rsp+E8h] [rbp-E8h]
  __m128i v52; // [rsp+F0h] [rbp-E0h] BYREF
  char v53; // [rsp+100h] [rbp-D0h]
  void *v54; // [rsp+108h] [rbp-C8h]
  size_t v55; // [rsp+110h] [rbp-C0h]
  __m128i v56; // [rsp+118h] [rbp-B8h] BYREF
  char v57; // [rsp+128h] [rbp-A8h]
  unsigned __int64 v58; // [rsp+130h] [rbp-A0h]
  void *s2; // [rsp+140h] [rbp-90h]
  size_t v60; // [rsp+148h] [rbp-88h]
  __m128i v61; // [rsp+150h] [rbp-80h] BYREF
  unsigned __int8 v62; // [rsp+160h] [rbp-70h]
  void *v63; // [rsp+168h] [rbp-68h]
  size_t v64; // [rsp+170h] [rbp-60h]
  __m128i v65; // [rsp+178h] [rbp-58h] BYREF
  unsigned __int8 v66; // [rsp+188h] [rbp-48h]
  unsigned __int64 v67; // [rsp+190h] [rbp-40h]

  sub_31185E0((__int64)&v46, *a1, *((_DWORD *)*a3 + 2));
  sub_31185E0((__int64)&v42, *a1, *((_DWORD *)*a3 + 3));
  v62 = 0;
  v5 = *a3;
  if ( v49 )
  {
    s2 = &v61;
    if ( v46 == &v48 )
    {
      v61 = _mm_load_si128(&v48);
    }
    else
    {
      s2 = v46;
      v61.m128i_i64[0] = v48.m128i_i64[0];
    }
    v6 = v47;
    v46 = &v48;
    v47 = 0;
    v60 = v6;
    v48.m128i_i8[0] = 0;
    v62 = 1;
  }
  v66 = 0;
  if ( v45 )
  {
    v63 = &v65;
    if ( v42 == &v44 )
    {
      v65 = _mm_load_si128(&v44);
    }
    else
    {
      v63 = v42;
      v65.m128i_i64[0] = v44.m128i_i64[0];
    }
    v7 = v43;
    v42 = &v44;
    v43 = 0;
    v64 = v7;
    v44.m128i_i8[0] = 0;
    v66 = 1;
  }
  v8 = *a1;
  v67 = *v5;
  sub_31185E0((__int64)&v38, v8, *((_DWORD *)*a2 + 2));
  sub_31185E0((__int64)&v34, *a1, *((_DWORD *)*a2 + 3));
  v53 = 0;
  v9 = *a2;
  if ( v41 )
  {
    s1 = &v52;
    if ( v38 == &v40 )
    {
      v52 = _mm_load_si128(&v40);
    }
    else
    {
      s1 = v38;
      v52.m128i_i64[0] = v40.m128i_i64[0];
    }
    v10 = v39;
    v38 = &v40;
    v39 = 0;
    n = v10;
    v40.m128i_i8[0] = 0;
    v53 = 1;
  }
  v11 = v37;
  v57 = 0;
  if ( !v37 )
  {
    v15 = *v9;
    v58 = v15;
    if ( v67 > v15 )
    {
      v11 = 1;
      goto LABEL_22;
    }
    if ( v67 != v15 || (v11 = v66) != 0 || !v62 || (v11 = v62, !v53) )
    {
LABEL_22:
      if ( !v53 )
        goto LABEL_23;
      goto LABEL_50;
    }
    v17 = 0;
    goto LABEL_35;
  }
  v54 = &v56;
  if ( v34 == &v36 )
  {
    v56 = _mm_load_si128(&v36);
  }
  else
  {
    v54 = v34;
    v56.m128i_i64[0] = v36.m128i_i64[0];
  }
  v12 = v35;
  v34 = &v36;
  v35 = 0;
  v55 = v12;
  v36.m128i_i8[0] = 0;
  v57 = 1;
  v13 = *v9;
  v58 = v13;
  if ( v13 < v67 )
  {
LABEL_46:
    v23 = (__m128i *)v54;
LABEL_47:
    v14 = v23;
    goto LABEL_48;
  }
  if ( v13 != v67 )
  {
    v11 = 0;
    goto LABEL_46;
  }
  v11 = v66;
  if ( !v66 )
  {
    v14 = (__m128i *)v54;
    goto LABEL_48;
  }
  v24 = (void *)v64;
  v23 = (__m128i *)v54;
  v25 = v63;
  v26 = v64;
  v14 = (__m128i *)v54;
  if ( v12 <= v64 )
    v26 = v12;
  if ( v26 )
  {
    v30 = (void *)v64;
    v31 = v26;
    v32 = v63;
    v27 = memcmp(v54, v63, v26);
    v25 = v32;
    v26 = v31;
    v24 = v30;
    if ( v27 )
    {
      if ( v27 < 0 )
        goto LABEL_48;
      goto LABEL_83;
    }
    v28 = v12 - (_QWORD)v30;
    if ( (__int64)(v12 - (_QWORD)v30) > 0x7FFFFFFF )
      goto LABEL_83;
LABEL_73:
    if ( v28 < (__int64)0xFFFFFFFF80000000LL || (int)v28 < 0 )
      goto LABEL_48;
    if ( !v26 )
      goto LABEL_76;
LABEL_83:
    v33 = v24;
    LODWORD(v29) = memcmp(v25, v23, v26);
    v24 = v33;
    if ( (_DWORD)v29 )
      goto LABEL_78;
    goto LABEL_76;
  }
  v28 = v12 - v64;
  if ( (__int64)(v12 - v64) <= 0x7FFFFFFF )
    goto LABEL_73;
LABEL_76:
  v29 = (signed __int64)v24 - v12;
  if ( (__int64)((__int64)v24 - v12) > 0x7FFFFFFF )
    goto LABEL_85;
  if ( v29 < (__int64)0xFFFFFFFF80000000LL )
  {
LABEL_79:
    v11 = 0;
    goto LABEL_48;
  }
LABEL_78:
  if ( (int)v29 < 0 )
    goto LABEL_79;
LABEL_85:
  v11 = v62;
  if ( v62 )
  {
    v17 = v53;
    if ( !v53 )
      goto LABEL_47;
LABEL_35:
    v18 = n;
    v19 = v60;
    v20 = v60;
    if ( n <= v60 )
      v20 = n;
    if ( !v20 || (v21 = memcmp(s1, s2, v20)) == 0 )
    {
      v22 = v18 - v19;
      v11 = 0;
      if ( v22 > 0x7FFFFFFF )
      {
LABEL_43:
        if ( !v17 )
          goto LABEL_22;
        v23 = (__m128i *)v54;
        goto LABEL_47;
      }
      if ( v22 < (__int64)0xFFFFFFFF80000000LL )
      {
        v11 = 1;
        goto LABEL_43;
      }
      v21 = v22;
    }
    v11 = v21 >> 31;
    goto LABEL_43;
  }
LABEL_48:
  v57 = 0;
  if ( v14 == &v56 )
    goto LABEL_22;
  j_j___libc_free_0((unsigned __int64)v14);
  if ( !v53 )
    goto LABEL_23;
LABEL_50:
  v53 = 0;
  if ( s1 == &v52 )
  {
LABEL_23:
    if ( !v37 )
      goto LABEL_24;
    goto LABEL_52;
  }
  j_j___libc_free_0((unsigned __int64)s1);
  if ( !v37 )
    goto LABEL_24;
LABEL_52:
  v37 = 0;
  if ( v34 == &v36 )
  {
LABEL_24:
    if ( !v41 )
      goto LABEL_25;
    goto LABEL_54;
  }
  j_j___libc_free_0((unsigned __int64)v34);
  if ( !v41 )
    goto LABEL_25;
LABEL_54:
  v41 = 0;
  if ( v38 == &v40 )
  {
LABEL_25:
    if ( !v66 )
      goto LABEL_26;
    goto LABEL_56;
  }
  j_j___libc_free_0((unsigned __int64)v38);
  if ( !v66 )
    goto LABEL_26;
LABEL_56:
  v66 = 0;
  if ( v63 == &v65 )
  {
LABEL_26:
    if ( !v62 )
      goto LABEL_27;
    goto LABEL_58;
  }
  j_j___libc_free_0((unsigned __int64)v63);
  if ( !v62 )
    goto LABEL_27;
LABEL_58:
  v62 = 0;
  if ( s2 == &v61 )
  {
LABEL_27:
    if ( !v45 )
      goto LABEL_28;
    goto LABEL_60;
  }
  j_j___libc_free_0((unsigned __int64)s2);
  if ( !v45 )
    goto LABEL_28;
LABEL_60:
  v45 = 0;
  if ( v42 == &v44 )
  {
LABEL_28:
    if ( !v49 )
      return v11;
    goto LABEL_62;
  }
  j_j___libc_free_0((unsigned __int64)v42);
  if ( !v49 )
    return v11;
LABEL_62:
  v49 = 0;
  if ( v46 != &v48 )
    j_j___libc_free_0((unsigned __int64)v46);
  return v11;
}
