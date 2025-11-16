// Function: sub_38943C0
// Address: 0x38943c0
//
__int64 __fastcall sub_38943C0(
        _QWORD *a1,
        unsigned int a2,
        unsigned __int64 *a3,
        unsigned __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v16; // r14
  __int64 v17; // rdi
  unsigned int v18; // r12d
  _BYTE *v21; // r8
  __int64 v22; // rax
  int *v23; // r9
  int *v24; // r12
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 *v27; // rdi
  __int64 v28; // r12
  __m128i *v29; // rax
  __int64 v30; // rcx
  __m128i *v31; // rax
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 *v35; // rdi
  __int64 v36; // r12
  __m128i *v37; // rax
  __m128i *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  double v41; // xmm4_8
  double v42; // xmm5_8
  int *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  double v46; // xmm4_8
  double v47; // xmm5_8
  int *v48; // rax
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // r14
  const char *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rbx
  __m128i *v54; // rax
  int *v55; // [rsp+8h] [rbp-108h]
  int *s2a; // [rsp+10h] [rbp-100h]
  void *s2; // [rsp+10h] [rbp-100h]
  size_t n; // [rsp+18h] [rbp-F8h]
  __int64 v59[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v60; // [rsp+30h] [rbp-E0h] BYREF
  __m128i *v61; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v62; // [rsp+48h] [rbp-C8h]
  __m128i v63; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v64[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v66; // [rsp+80h] [rbp-90h] BYREF
  __int64 v67; // [rsp+88h] [rbp-88h]
  __m128i v68; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int64 **v69; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 *v70; // [rsp+A8h] [rbp-68h]
  __int64 v71; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 *v72; // [rsp+C0h] [rbp-50h] BYREF
  char *v73; // [rsp+C8h] [rbp-48h]
  _OWORD v74[4]; // [rsp+D0h] [rbp-40h] BYREF

  v16 = *(_QWORD *)a5;
  if ( !*(_BYTE *)(*(_QWORD *)a5 + 8LL) )
  {
    if ( a2 != -1 || a3[1] )
    {
      v17 = *a1;
      v72 = (unsigned __int64 *)"instructions returning void cannot have a name";
      LOWORD(v74[0]) = 259;
      return (unsigned int)sub_38814C0(v17 + 8, a4, (__int64)&v72);
    }
    return 0;
  }
  if ( !a3[1] )
  {
    v21 = (_BYTE *)a1[15];
    if ( a2 == -1 )
      a2 = (__int64)&v21[-a1[14]] >> 3;
    if ( a2 != (__int64)&v21[-a1[14]] >> 3 )
    {
      v66 = (__int64)(a1[15] - a1[14]) >> 3;
      v17 = *a1;
      v69 = (unsigned __int64 **)"instruction expected to be numbered '%";
      v70 = &v66;
      v72 = (unsigned __int64 *)&v69;
      LOWORD(v71) = 2819;
      v73 = "'";
      LOWORD(v74[0]) = 770;
      return (unsigned int)sub_38814C0(v17 + 8, a4, (__int64)&v72);
    }
    v22 = a1[10];
    v23 = (int *)(a1 + 9);
    if ( v22 )
    {
      v24 = (int *)(a1 + 9);
      do
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)(v22 + 16);
          v26 = *(_QWORD *)(v22 + 24);
          if ( a2 <= *(_DWORD *)(v22 + 32) )
            break;
          v22 = *(_QWORD *)(v22 + 24);
          if ( !v26 )
            goto LABEL_16;
        }
        v24 = (int *)v22;
        v22 = *(_QWORD *)(v22 + 16);
      }
      while ( v25 );
LABEL_16:
      if ( v23 != v24 && a2 >= v24[8] )
      {
        v27 = (__int64 *)*((_QWORD *)v24 + 5);
        if ( v16 != *v27 )
        {
          v28 = *a1;
          sub_3888960(v59, *v27);
          v29 = (__m128i *)sub_2241130(
                             (unsigned __int64 *)v59,
                             0,
                             0,
                             "instruction forward referenced with type '",
                             0x2Au);
          v61 = &v63;
          if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
          {
            v63 = _mm_loadu_si128(v29 + 1);
          }
          else
          {
            v61 = (__m128i *)v29->m128i_i64[0];
            v63.m128i_i64[0] = v29[1].m128i_i64[0];
          }
          v30 = v29->m128i_i64[1];
          v29[1].m128i_i8[0] = 0;
          v62 = v30;
          v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
          v29->m128i_i64[1] = 0;
          if ( v62 != 0x3FFFFFFFFFFFFFFFLL )
          {
            v31 = (__m128i *)sub_2241490((unsigned __int64 *)&v61, "'", 1u);
            v72 = (unsigned __int64 *)v74;
            if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
            {
              v74[0] = _mm_loadu_si128(v31 + 1);
            }
            else
            {
              v72 = (unsigned __int64 *)v31->m128i_i64[0];
              *(_QWORD *)&v74[0] = v31[1].m128i_i64[0];
            }
            v73 = (char *)v31->m128i_i64[1];
            v31->m128i_i64[0] = (__int64)v31[1].m128i_i64;
            v31->m128i_i64[1] = 0;
            v31[1].m128i_i8[0] = 0;
            LOWORD(v71) = 260;
            v69 = &v72;
            v18 = sub_38814C0(v28 + 8, a4, (__int64)&v69);
            if ( v72 != (unsigned __int64 *)v74 )
              j_j___libc_free_0((unsigned __int64)v72);
            if ( v61 != &v63 )
              j_j___libc_free_0((unsigned __int64)v61);
            if ( (__int64 *)v59[0] != &v60 )
              j_j___libc_free_0(v59[0]);
            return v18;
          }
LABEL_70:
          sub_4262D8((__int64)"basic_string::append");
        }
        s2a = v23;
        sub_164D160((__int64)v27, a5, a6, a7, a8, a9, a10, a11, a12, a13);
        sub_164BEC0((__int64)v27, a5, v39, v40, a6, a7, a8, a9, v41, v42, a12, a13);
        v43 = sub_220F330(v24, s2a);
        j_j___libc_free_0((unsigned __int64)v43);
        --a1[13];
        v21 = (_BYTE *)a1[15];
      }
    }
    v72 = (unsigned __int64 *)a5;
    if ( v21 == (_BYTE *)a1[16] )
    {
      sub_12879C0((__int64)(a1 + 14), v21, &v72);
    }
    else
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = a5;
        v21 = (_BYTE *)a1[15];
      }
      a1[15] = v21 + 8;
    }
    return 0;
  }
  v32 = sub_3890600((__int64)(a1 + 2), (__int64)a3);
  if ( (_QWORD *)v32 != a1 + 3 )
  {
    v35 = *(__int64 **)(v32 + 64);
    if ( v16 != *v35 )
    {
      v36 = *a1;
      sub_3888960(v64, *v35);
      v37 = (__m128i *)sub_2241130((unsigned __int64 *)v64, 0, 0, "instruction forward referenced with type '", 0x2Au);
      v66 = (unsigned __int64)&v68;
      if ( (__m128i *)v37->m128i_i64[0] == &v37[1] )
      {
        v68 = _mm_loadu_si128(v37 + 1);
      }
      else
      {
        v66 = v37->m128i_i64[0];
        v68.m128i_i64[0] = v37[1].m128i_i64[0];
      }
      v67 = v37->m128i_i64[1];
      v37->m128i_i64[0] = (__int64)v37[1].m128i_i64;
      v37->m128i_i64[1] = 0;
      v37[1].m128i_i8[0] = 0;
      if ( v67 == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_70;
      v38 = (__m128i *)sub_2241490(&v66, "'", 1u);
      v72 = (unsigned __int64 *)v74;
      if ( (__m128i *)v38->m128i_i64[0] == &v38[1] )
      {
        v74[0] = _mm_loadu_si128(v38 + 1);
      }
      else
      {
        v72 = (unsigned __int64 *)v38->m128i_i64[0];
        *(_QWORD *)&v74[0] = v38[1].m128i_i64[0];
      }
      v73 = (char *)v38->m128i_i64[1];
      v38->m128i_i64[0] = (__int64)v38[1].m128i_i64;
      v38->m128i_i64[1] = 0;
      v38[1].m128i_i8[0] = 0;
      LOWORD(v71) = 260;
      v69 = &v72;
      v18 = sub_38814C0(v36 + 8, a4, (__int64)&v69);
      if ( v72 != (unsigned __int64 *)v74 )
        j_j___libc_free_0((unsigned __int64)v72);
      if ( (__m128i *)v66 != &v68 )
        j_j___libc_free_0(v66);
      if ( (__int64 *)v64[0] != &v65 )
        j_j___libc_free_0(v64[0]);
      return v18;
    }
    v55 = (int *)v32;
    sub_164D160((__int64)v35, a5, a6, a7, a8, a9, v33, v34, a12, a13);
    sub_164BEC0((__int64)v35, a5, v44, v45, a6, a7, a8, a9, v46, v47, a12, a13);
    v48 = sub_220F330(v55, a1 + 3);
    v49 = *((_QWORD *)v48 + 4);
    v50 = (unsigned __int64)v48;
    if ( (int *)v49 != v48 + 12 )
      j_j___libc_free_0(v49);
    j_j___libc_free_0(v50);
    --a1[7];
  }
  v72 = a3;
  LOWORD(v74[0]) = 260;
  sub_164B780(a5, (__int64 *)&v72);
  n = a3[1];
  s2 = (void *)*a3;
  v51 = sub_1649960(a5);
  if ( n == v52 && (!n || !memcmp(v51, s2, n)) )
    return 0;
  v53 = *a1;
  sub_8FD6D0((__int64)&v69, "multiple definition of local value named '", a3);
  if ( v70 == (unsigned __int64 *)0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_70;
  v54 = (__m128i *)sub_2241490((unsigned __int64 *)&v69, "'", 1u);
  v72 = (unsigned __int64 *)v74;
  if ( (__m128i *)v54->m128i_i64[0] == &v54[1] )
  {
    v74[0] = _mm_loadu_si128(v54 + 1);
  }
  else
  {
    v72 = (unsigned __int64 *)v54->m128i_i64[0];
    *(_QWORD *)&v74[0] = v54[1].m128i_i64[0];
  }
  v73 = (char *)v54->m128i_i64[1];
  v54->m128i_i64[0] = (__int64)v54[1].m128i_i64;
  v54->m128i_i64[1] = 0;
  v54[1].m128i_i8[0] = 0;
  v68.m128i_i16[0] = 260;
  v66 = (unsigned __int64)&v72;
  v18 = sub_38814C0(v53 + 8, a4, (__int64)&v66);
  if ( v72 != (unsigned __int64 *)v74 )
    j_j___libc_free_0((unsigned __int64)v72);
  if ( v69 != (unsigned __int64 **)&v71 )
    j_j___libc_free_0((unsigned __int64)v69);
  return v18;
}
