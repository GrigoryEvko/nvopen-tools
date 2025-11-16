// Function: sub_36FFE70
// Address: 0x36ffe70
//
void __fastcall sub_36FFE70(__int64 *a1, _BYTE *a2, unsigned __int64 a3)
{
  size_t v6; // rax
  __int64 *v7; // rdx
  __int64 *v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rcx
  size_t v11; // rsi
  __int64 v12; // rdi
  size_t v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 *v18; // rdi
  _QWORD *v19; // rdi
  __int64 v20; // rsi
  size_t v21; // rdx
  void *dest; // [rsp+10h] [rbp-180h] BYREF
  size_t v23; // [rsp+18h] [rbp-178h]
  __m128i v24; // [rsp+20h] [rbp-170h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-160h] BYREF
  size_t n; // [rsp+38h] [rbp-158h]
  _QWORD src[2]; // [rsp+40h] [rbp-150h] BYREF
  unsigned __int16 v28; // [rsp+50h] [rbp-140h] BYREF
  _BYTE *v29; // [rsp+58h] [rbp-138h]
  unsigned __int64 v30; // [rsp+60h] [rbp-130h]
  size_t v31; // [rsp+80h] [rbp-110h] BYREF
  __m128i *v32; // [rsp+88h] [rbp-108h]
  size_t v33; // [rsp+90h] [rbp-100h]
  __m128i v34; // [rsp+98h] [rbp-F8h] BYREF
  __int64 *v35; // [rsp+B0h] [rbp-E0h] BYREF
  size_t v36; // [rsp+B8h] [rbp-D8h]
  _QWORD v37[26]; // [rsp+C0h] [rbp-D0h] BYREF

  if ( !a2 )
  {
    LOBYTE(v37[0]) = 0;
    v8 = (__int64 *)a1[18];
    v13 = 0;
    v35 = v37;
LABEL_10:
    a1[19] = v13;
    *((_BYTE *)v8 + v13) = 0;
    v9 = v35;
    goto LABEL_11;
  }
  v31 = a3;
  v6 = a3;
  v35 = v37;
  if ( a3 > 0xF )
  {
    v35 = (__int64 *)sub_22409D0((__int64)&v35, &v31, 0);
    v18 = v35;
    v37[0] = v31;
LABEL_21:
    memcpy(v18, a2, a3);
    v6 = v31;
    v7 = v35;
    goto LABEL_5;
  }
  if ( a3 == 1 )
  {
    LOBYTE(v37[0]) = *a2;
    v7 = v37;
    goto LABEL_5;
  }
  if ( a3 )
  {
    v18 = v37;
    goto LABEL_21;
  }
  v7 = v37;
LABEL_5:
  v36 = v6;
  *((_BYTE *)v7 + v6) = 0;
  v8 = (__int64 *)a1[18];
  v9 = v8;
  if ( v35 == v37 )
  {
    v13 = v36;
    if ( v36 )
    {
      if ( v36 == 1 )
        *(_BYTE *)v8 = v37[0];
      else
        memcpy(v8, v37, v36);
      v13 = v36;
      v8 = (__int64 *)a1[18];
    }
    goto LABEL_10;
  }
  v10 = v37[0];
  v11 = v36;
  if ( v8 == a1 + 20 )
  {
    a1[18] = (__int64)v35;
    a1[19] = v11;
    a1[20] = v10;
  }
  else
  {
    v12 = a1[20];
    a1[18] = (__int64)v35;
    a1[19] = v11;
    a1[20] = v10;
    if ( v9 )
    {
      v35 = v9;
      v37[0] = v12;
      goto LABEL_11;
    }
  }
  v35 = v37;
  v9 = v37;
LABEL_11:
  v36 = 0;
  *(_BYTE *)v9 = 0;
  if ( v35 != v37 )
    j_j___libc_free_0((unsigned __int64)v35);
  v14 = *a1;
  v35 = v37;
  v37[16] = 0;
  v37[18] = v14;
  v36 = 0x1000000001LL;
  v37[17] = 0;
  v37[19] = 0;
  LODWORD(v37[0]) = 0;
  BYTE4(v37[0]) = 0;
  sub_C6ACB0((__int64)&v35);
  v28 = 5;
  v29 = a2;
  v30 = a3;
  if ( !(unsigned __int8)sub_C6A630(a2, a3, 0) )
  {
    sub_C6B0E0((__int64 *)&dest, (__int64)a2, a3);
    LOWORD(v31) = 6;
    if ( (unsigned __int8)sub_C6A630((char *)dest, v23, 0) )
    {
LABEL_27:
      v32 = &v34;
      if ( dest == &v24 )
      {
        v34 = _mm_load_si128(&v24);
      }
      else
      {
        v32 = (__m128i *)dest;
        v34.m128i_i64[0] = v24.m128i_i64[0];
      }
      dest = &v24;
      v33 = v23;
      v23 = 0;
      v24.m128i_i8[0] = 0;
      sub_C6BC50(&v28);
      sub_C6A4F0((__int64)&v28, (unsigned __int16 *)&v31);
      sub_C6BC50((unsigned __int16 *)&v31);
      if ( dest != &v24 )
        j_j___libc_free_0((unsigned __int64)dest);
      goto LABEL_14;
    }
    sub_C6B0E0((__int64 *)&v25, (__int64)dest, v23);
    v19 = dest;
    if ( v25 == src )
    {
      v21 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v21 = n;
        v19 = dest;
      }
      v23 = v21;
      *((_BYTE *)v19 + v21) = 0;
      v19 = v25;
      goto LABEL_36;
    }
    if ( dest == &v24 )
    {
      dest = v25;
      v23 = n;
      v24.m128i_i64[0] = src[0];
    }
    else
    {
      v20 = v24.m128i_i64[0];
      dest = v25;
      v23 = n;
      v24.m128i_i64[0] = src[0];
      if ( v19 )
      {
        v25 = v19;
        src[0] = v20;
        goto LABEL_36;
      }
    }
    v25 = src;
    v19 = src;
LABEL_36:
    n = 0;
    *(_BYTE *)v19 = 0;
    if ( v25 != src )
      j_j___libc_free_0((unsigned __int64)v25);
    goto LABEL_27;
  }
LABEL_14:
  sub_C6B410((__int64)&v35, (unsigned __int8 *)"context", 7u);
  sub_C6C710((__int64)&v35, &v28, v15);
  sub_C6AE10((__int64)&v35);
  sub_C6BC50(&v28);
  sub_C6AD90((__int64)&v35);
  v16 = *a1;
  v17 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == v17 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v17 = 10;
    ++*(_QWORD *)(v16 + 32);
  }
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
}
