// Function: sub_2285FB0
// Address: 0x2285fb0
//
__int64 __fastcall sub_2285FB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 (__fastcall *a4)(unsigned __int64 *),
        __int64 a5)
{
  __int64 v6; // r12
  bool v11; // zf
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // r14
  double v16; // xmm0_8
  __int64 v17; // rax
  double v18; // xmm1_8
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  int v22; // eax
  _BYTE *v23; // rsi
  int v24; // ecx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  char v27; // r10
  __int64 v28; // r9
  __m128i *v29; // rax
  __int64 v30; // rcx
  __m128i *v31; // rax
  size_t v32; // rcx
  __m128i *v33; // r10
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdi
  __m128i *v36; // rax
  __m128i *v37; // rcx
  __m128i *v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  __m128i *v41; // [rsp+10h] [rbp-E0h]
  __int64 v42; // [rsp+18h] [rbp-D8h]
  __m128i v43; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD *v44; // [rsp+30h] [rbp-C0h] BYREF
  int v45; // [rsp+38h] [rbp-B8h]
  _QWORD v46[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i *v47; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-98h]
  __m128i v49; // [rsp+60h] [rbp-90h] BYREF
  __m128i *v50; // [rsp+70h] [rbp-80h] BYREF
  size_t v51; // [rsp+78h] [rbp-78h]
  __m128i v52; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v53; // [rsp+90h] [rbp-60h] BYREF
  size_t v54; // [rsp+98h] [rbp-58h]
  unsigned __int64 v55; // [rsp+A0h] [rbp-50h] BYREF
  char v56; // [rsp+A8h] [rbp-48h]
  __int64 v57; // [rsp+B0h] [rbp-40h]

  if ( !(_BYTE)qword_4FDB228 )
    goto LABEL_4;
  v6 = *(_QWORD *)(a2 + 8);
  if ( !v6 || sub_B2FC80(*(_QWORD *)(a2 + 8)) )
    goto LABEL_4;
  v11 = *(_BYTE *)(a3 + 24) == 0;
  v56 = 0;
  if ( !v11 )
  {
    v39 = *(_QWORD *)(a3 + 16);
    v53 = 6;
    v54 = 0;
    v55 = v39;
    if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
      sub_BD6050(&v53, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
    v56 = 1;
  }
  v57 = *(_QWORD *)(a3 + 32);
  v12 = a4(&v53);
  if ( v56 )
  {
    v56 = 0;
    if ( v55 != -4096 && v55 != 0 && v55 != -8192 )
      sub_BD60C0(&v53);
  }
  v13 = *(_QWORD *)(v12 + 8);
  if ( !v13 )
  {
LABEL_4:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v14 = sub_11FCB90(v6, v13);
  v15 = v14;
  if ( v14 < 0 )
    v16 = (double)(int)(v14 & 1 | ((unsigned __int64)v14 >> 1)) + (double)(int)(v14 & 1 | ((unsigned __int64)v14 >> 1));
  else
    v16 = (double)(int)v14;
  v17 = *(_QWORD *)(a5 + 48);
  if ( v17 < 0 )
  {
    v40 = *(_QWORD *)(a5 + 48) & 1LL | (*(_QWORD *)(a5 + 48) >> 1);
    v18 = (double)(int)v40 + (double)(int)v40;
  }
  else
  {
    v18 = (double)(int)v17;
  }
  sub_11F4620(
    (__int64 *)&v53,
    (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
    328,
    (__int64)"%f",
    v16 / v18 + v16 / v18 + 1.0);
  if ( v15 <= 9 )
  {
    v44 = v46;
    sub_2240A50((__int64 *)&v44, 1u, 0);
    v23 = v44;
LABEL_26:
    *v23 = v15 + 48;
    goto LABEL_27;
  }
  if ( v15 <= 0x63 )
  {
    v44 = v46;
    sub_2240A50((__int64 *)&v44, 2u, 0);
    v23 = v44;
  }
  else
  {
    if ( v15 <= 0x3E7 )
    {
      v20 = 3;
    }
    else if ( v15 <= 0x270F )
    {
      v20 = 4;
    }
    else
    {
      v19 = v15;
      LODWORD(v20) = 1;
      while ( 1 )
      {
        v21 = v19;
        v22 = v20;
        v20 = (unsigned int)(v20 + 4);
        v19 /= 0x2710u;
        if ( v21 <= 0x1869F )
          break;
        if ( v21 <= 0xF423F )
        {
          v44 = v46;
          v20 = (unsigned int)(v22 + 5);
          goto LABEL_23;
        }
        if ( v21 <= (unsigned __int64)&loc_98967F )
        {
          v20 = (unsigned int)(v22 + 6);
          break;
        }
        if ( v21 <= 0x5F5E0FF )
        {
          v20 = (unsigned int)(v22 + 7);
          break;
        }
      }
    }
    v44 = v46;
LABEL_23:
    sub_2240A50((__int64 *)&v44, v20, 0);
    v23 = v44;
    v24 = v45 - 1;
    do
    {
      v25 = v15
          - 20 * (v15 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v15 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v26 = v15;
      v15 /= 0x64u;
      v27 = a00010203040506_0[2 * v25 + 1];
      LOBYTE(v25) = a00010203040506_0[2 * v25];
      v23[v24] = v27;
      v28 = (unsigned int)(v24 - 1);
      v24 -= 2;
      v23[v28] = v25;
    }
    while ( v26 > 0x270F );
    if ( v26 <= 0x3E7 )
      goto LABEL_26;
  }
  v23[1] = a00010203040506_0[2 * v15 + 1];
  *v23 = a00010203040506_0[2 * v15];
LABEL_27:
  v29 = (__m128i *)sub_2241130((unsigned __int64 *)&v44, 0, 0, "label=\"", 7u);
  v47 = &v49;
  if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
  {
    v49 = _mm_loadu_si128(v29 + 1);
  }
  else
  {
    v47 = (__m128i *)v29->m128i_i64[0];
    v49.m128i_i64[0] = v29[1].m128i_i64[0];
  }
  v30 = v29->m128i_i64[1];
  v29[1].m128i_i8[0] = 0;
  v48 = v30;
  v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
  v29->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v48) <= 0xA )
    sub_4262D8((__int64)"basic_string::append");
  v31 = (__m128i *)sub_2241490((unsigned __int64 *)&v47, "\" penwidth=", 0xBu);
  v50 = &v52;
  if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
  {
    v52 = _mm_loadu_si128(v31 + 1);
  }
  else
  {
    v50 = (__m128i *)v31->m128i_i64[0];
    v52.m128i_i64[0] = v31[1].m128i_i64[0];
  }
  v32 = v31->m128i_u64[1];
  v31[1].m128i_i8[0] = 0;
  v51 = v32;
  v31->m128i_i64[0] = (__int64)v31[1].m128i_i64;
  v33 = v50;
  v31->m128i_i64[1] = 0;
  v34 = 15;
  v35 = 15;
  if ( v33 != &v52 )
    v35 = v52.m128i_i64[0];
  if ( v51 + v54 > v35 )
  {
    if ( (unsigned __int64 *)v53 != &v55 )
      v34 = v55;
    if ( v51 + v54 <= v34 )
    {
      v36 = (__m128i *)sub_2241130(&v53, 0, 0, v33, v51);
      v41 = &v43;
      v37 = (__m128i *)v36->m128i_i64[0];
      v38 = v36 + 1;
      if ( (__m128i *)v36->m128i_i64[0] != &v36[1] )
        goto LABEL_39;
LABEL_61:
      v43 = _mm_loadu_si128(v36 + 1);
      goto LABEL_40;
    }
  }
  v36 = (__m128i *)sub_2241490((unsigned __int64 *)&v50, (char *)v53, v54);
  v41 = &v43;
  v37 = (__m128i *)v36->m128i_i64[0];
  v38 = v36 + 1;
  if ( (__m128i *)v36->m128i_i64[0] == &v36[1] )
    goto LABEL_61;
LABEL_39:
  v41 = v37;
  v43.m128i_i64[0] = v36[1].m128i_i64[0];
LABEL_40:
  v42 = v36->m128i_i64[1];
  v36->m128i_i64[0] = (__int64)v38;
  v36->m128i_i64[1] = 0;
  v36[1].m128i_i8[0] = 0;
  if ( v50 != &v52 )
    j_j___libc_free_0((unsigned __int64)v50);
  if ( v47 != &v49 )
    j_j___libc_free_0((unsigned __int64)v47);
  if ( v44 != v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( (unsigned __int64 *)v53 != &v55 )
    j_j___libc_free_0(v53);
  *(_QWORD *)a1 = a1 + 16;
  if ( v41 == &v43 )
  {
    *(__m128i *)(a1 + 16) = _mm_load_si128(&v43);
  }
  else
  {
    *(_QWORD *)a1 = v41;
    *(_QWORD *)(a1 + 16) = v43.m128i_i64[0];
  }
  *(_QWORD *)(a1 + 8) = v42;
  return a1;
}
