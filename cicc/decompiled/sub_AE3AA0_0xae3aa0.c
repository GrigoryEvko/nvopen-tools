// Function: sub_AE3AA0
// Address: 0xae3aa0
//
unsigned __int64 *__fastcall sub_AE3AA0(unsigned __int64 *a1, _QWORD *a2, _BYTE *a3, __int64 a4)
{
  size_t *v8; // rdi
  size_t v9; // rsi
  size_t v10; // r8
  size_t v11; // rdx
  __int64 v13; // rsi
  char *v14; // rdx
  __int64 v15; // r9
  __m128i v16; // xmm2
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r13
  unsigned int *v25; // rdi
  char *v26; // rdx
  unsigned int *v27; // r13
  unsigned int *v28; // r14
  unsigned int v29; // r15d
  _DWORD *v30; // rax
  size_t v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // r13
  unsigned int v35; // ebx
  __int64 v36; // [rsp+10h] [rbp-180h]
  char *src; // [rsp+18h] [rbp-178h]
  __int64 v38[2]; // [rsp+20h] [rbp-170h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-160h] BYREF
  unsigned int *v40; // [rsp+40h] [rbp-150h] BYREF
  __int64 v41; // [rsp+48h] [rbp-148h]
  _BYTE v42[32]; // [rsp+50h] [rbp-140h] BYREF
  char v43; // [rsp+70h] [rbp-120h] BYREF
  __m128i v44; // [rsp+78h] [rbp-118h]
  __m128i v45; // [rsp+88h] [rbp-108h] BYREF
  __m128i v46; // [rsp+98h] [rbp-F8h]
  char v47; // [rsp+B0h] [rbp-E0h] BYREF
  __m128i v48; // [rsp+B8h] [rbp-D8h]
  __m128i v49; // [rsp+C8h] [rbp-C8h]
  __m128i v50; // [rsp+D8h] [rbp-B8h]
  size_t *v51; // [rsp+F0h] [rbp-A0h] BYREF
  size_t n[2]; // [rsp+F8h] [rbp-98h] BYREF
  __m128i v53; // [rsp+108h] [rbp-88h] BYREF
  __m128i v54; // [rsp+118h] [rbp-78h] BYREF
  char v55; // [rsp+128h] [rbp-68h] BYREF
  __m128i v56; // [rsp+130h] [rbp-60h] BYREF
  __m128i v57; // [rsp+140h] [rbp-50h] BYREF
  __m128i v58; // [rsp+150h] [rbp-40h] BYREF

  v51 = &n[1];
  sub_AE11D0((__int64 *)&v51, a3, (__int64)&a3[a4]);
  v8 = (size_t *)a2[56];
  if ( v51 == &n[1] )
  {
    v31 = n[0];
    if ( n[0] )
    {
      if ( n[0] == 1 )
        *(_BYTE *)v8 = n[1];
      else
        memcpy(v8, &n[1], n[0]);
      v31 = n[0];
      v8 = (size_t *)a2[56];
    }
    a2[57] = v31;
    *((_BYTE *)v8 + v31) = 0;
    v8 = v51;
  }
  else
  {
    v9 = n[1];
    v10 = n[0];
    if ( v8 == a2 + 58 )
    {
      a2[56] = v51;
      a2[57] = v10;
      a2[58] = v9;
    }
    else
    {
      v11 = a2[58];
      a2[56] = v51;
      a2[57] = v10;
      a2[58] = v9;
      if ( v8 )
      {
        v51 = v8;
        n[1] = v11;
        goto LABEL_5;
      }
    }
    v51 = &n[1];
    v8 = &n[1];
  }
LABEL_5:
  n[0] = 0;
  *(_BYTE *)v8 = 0;
  if ( v51 != &n[1] )
    j_j___libc_free_0(v51, n[1] + 1);
  if ( a4 )
  {
    v13 = (__int64)a3;
    v40 = (unsigned int *)v42;
    v41 = 0x800000000LL;
    sub_AE1A40((__int64)&v51, (__int64)a3, a4, 45);
    v16 = _mm_loadu_si128(&v54);
    v17 = _mm_loadu_si128((const __m128i *)n);
    v18 = _mm_loadu_si128(&v53);
    v43 = (char)v51;
    v44 = v17;
    v45 = v18;
    v46 = v16;
    if ( (size_t **)v54.m128i_i64[0] == &v51 )
    {
      v46.m128i_i64[1] = 1;
      v46.m128i_i64[0] = (__int64)&v43;
    }
    v19 = _mm_loadu_si128(&v56);
    v20 = _mm_loadu_si128(&v57);
    v21 = _mm_loadu_si128(&v58);
    v47 = v55;
    v48 = v19;
    v49 = v20;
    v50 = v21;
    if ( (char *)v58.m128i_i64[0] == &v55 )
    {
      v50.m128i_i64[1] = 1;
      v50.m128i_i64[0] = (__int64)&v47;
    }
    v36 = v48.m128i_i64[0];
    v22 = v48.m128i_i64[0];
    v23 = v44.m128i_u64[1];
    if ( v44.m128i_i64[0] == v48.m128i_i64[0] )
    {
LABEL_29:
      v25 = v40;
      v27 = &v40[(unsigned int)v41];
      v28 = v40;
      if ( v27 != v40 )
      {
        do
        {
          v29 = *v28++;
          v30 = sub_AE2980((__int64)a2, v29);
          v13 = v29;
          sub_AE2A10((__int64)a2, v29, v30[1], *((_BYTE *)v30 + 8), *((_BYTE *)v30 + 9), v30[3], 1u);
        }
        while ( v27 != v28 );
        v25 = v40;
      }
      *a1 = 1;
    }
    else
    {
      while ( v23 )
      {
        v13 = (__int64)a2;
        sub_AE30B0((unsigned __int64 *)v38, (__int64)a2, v44.m128i_i64[0], v44.m128i_u64[1], (__int64)&v40, v15);
        v24 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = v24 | 1;
          goto LABEL_20;
        }
        src = (char *)v46.m128i_i64[1];
        v13 = v46.m128i_i64[0];
        v23 = sub_C931B0(&v45, v46.m128i_i64[0], v46.m128i_i64[1], 0);
        if ( v23 == -1 )
        {
          v23 = v45.m128i_u64[1];
          v22 = v45.m128i_i64[0];
          v14 = 0;
        }
        else
        {
          v13 = v45.m128i_i64[1];
          v22 = v45.m128i_i64[0];
          v26 = &src[v23];
          if ( (unsigned __int64)&src[v23] > v45.m128i_i64[1] )
            v26 = (char *)v45.m128i_i64[1];
          else
            v24 = v45.m128i_i64[1] - (_QWORD)v26;
          v14 = &v26[v45.m128i_i64[0]];
          if ( v23 > v45.m128i_i64[1] )
            v23 = v45.m128i_u64[1];
        }
        v44.m128i_i64[0] = v22;
        v44.m128i_i64[1] = v23;
        v45.m128i_i64[0] = (__int64)v14;
        v45.m128i_i64[1] = v24;
        if ( v36 == v22 )
          goto LABEL_29;
      }
      v32 = sub_C63BB0(v44.m128i_i64[1], v44.m128i_i64[0], v14, v22);
      v34 = v33;
      v35 = v32;
      v38[0] = (__int64)v39;
      sub_AE11D0(v38, "empty specification is not allowed", (__int64)"");
      v13 = (__int64)v38;
      sub_C63F00(a1, v38, v35, v34);
      if ( (_QWORD *)v38[0] != v39 )
      {
        v13 = v39[0] + 1LL;
        j_j___libc_free_0(v38[0], v39[0] + 1LL);
      }
LABEL_20:
      v25 = v40;
    }
    if ( v25 != (unsigned int *)v42 )
      _libc_free(v25, v13);
  }
  else
  {
    *a1 = 1;
  }
  return a1;
}
