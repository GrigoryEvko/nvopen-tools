// Function: sub_CCA5D0
// Address: 0xcca5d0
//
__int64 __fastcall sub_CCA5D0(__int64 a1, _BYTE *a2, _BYTE *a3, _BYTE *a4, _BYTE *a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r10
  void *v8; // r11
  void *v9; // r12
  char v13; // dl
  void *v14; // rcx
  char v15; // cl
  void **v16; // rdi
  __int64 v17; // rsi
  __m128i *v18; // rcx
  char v19; // cl
  char v20; // dl
  __int64 v21; // rsi
  __m128i *v22; // rcx
  char v23; // al
  __int64 v24; // rcx
  int v25; // eax
  __m128i *v26; // rdi
  int v27; // eax
  __m128i *v28; // rdi
  __m128i *v29; // rdi
  __m128i *v30; // rdi
  __m128i *v31; // rdi
  __m128i *v32; // rdi
  __int64 result; // rax
  __int64 v34; // [rsp+0h] [rbp-180h]
  __int64 v35; // [rsp+8h] [rbp-178h]
  __int64 v36; // [rsp+10h] [rbp-170h]
  __int64 v37; // [rsp+18h] [rbp-168h]
  __m128i v40; // [rsp+30h] [rbp-150h] BYREF
  __m128i v41; // [rsp+40h] [rbp-140h]
  __int64 v42; // [rsp+50h] [rbp-130h]
  __m128i v43; // [rsp+60h] [rbp-120h] BYREF
  __m128i v44; // [rsp+70h] [rbp-110h] BYREF
  __int64 v45; // [rsp+80h] [rbp-100h]
  __m128i v46; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v47; // [rsp+A0h] [rbp-E0h]
  __int64 v48; // [rsp+B0h] [rbp-D0h]
  __m128i v49; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v50; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v51; // [rsp+E0h] [rbp-A0h]
  __m128i v52; // [rsp+F0h] [rbp-90h] BYREF
  __m128i v53; // [rsp+100h] [rbp-80h]
  __int64 v54; // [rsp+110h] [rbp-70h]
  void *s1[2]; // [rsp+120h] [rbp-60h] BYREF
  __m128i v56; // [rsp+130h] [rbp-50h] BYREF
  __int64 v57; // [rsp+140h] [rbp-40h]

  v13 = a2[32];
  if ( !v13 )
  {
    LOWORD(v57) = 256;
LABEL_28:
    LOWORD(v54) = 256;
    LOWORD(v51) = 256;
LABEL_29:
    LOWORD(v48) = 256;
    LOWORD(v45) = 256;
LABEL_30:
    LOWORD(v42) = 256;
    goto LABEL_31;
  }
  if ( v13 == 1 )
  {
    LOBYTE(s1[0]) = 45;
    v13 = 8;
    LOWORD(v57) = 264;
  }
  else
  {
    if ( a2[33] == 1 )
    {
      v9 = (void *)*((_QWORD *)a2 + 1);
      v14 = *(void **)a2;
    }
    else
    {
      v14 = a2;
      v13 = 2;
    }
    s1[0] = v14;
    s1[1] = v9;
    v56.m128i_i8[0] = 45;
    LOBYTE(v57) = v13;
    BYTE1(v57) = 8;
  }
  v15 = a3[32];
  if ( !v15 )
    goto LABEL_28;
  if ( v15 == 1 )
  {
    v13 = v57;
    v52 = _mm_loadu_si128((const __m128i *)s1);
    v54 = v57;
    v53 = _mm_loadu_si128(&v56);
  }
  else
  {
    if ( BYTE1(v57) == 1 )
    {
      v8 = s1[1];
      v16 = (void **)s1[0];
    }
    else
    {
      v13 = 2;
      v16 = s1;
    }
    if ( a3[33] == 1 )
    {
      v17 = *(_QWORD *)a3;
      v37 = *((_QWORD *)a3 + 1);
    }
    else
    {
      v17 = (__int64)a3;
      v15 = 2;
    }
    v53.m128i_i64[0] = v17;
    v52.m128i_i64[0] = (__int64)v16;
    v52.m128i_i64[1] = (__int64)v8;
    v53.m128i_i64[1] = v37;
    LOBYTE(v54) = v13;
    BYTE1(v54) = v15;
  }
  if ( BYTE1(v54) == 1 )
  {
    v7 = v52.m128i_i64[1];
    v18 = (__m128i *)v52.m128i_i64[0];
  }
  else
  {
    v18 = &v52;
    v13 = 2;
  }
  v49.m128i_i64[0] = (__int64)v18;
  v49.m128i_i64[1] = v7;
  v50.m128i_i8[0] = 45;
  LOBYTE(v51) = v13;
  BYTE1(v51) = 8;
  v19 = a4[32];
  if ( !v19 )
    goto LABEL_29;
  if ( v19 == 1 )
  {
    v20 = v51;
    v46 = _mm_loadu_si128(&v49);
    v48 = v51;
    v47 = _mm_loadu_si128(&v50);
  }
  else
  {
    v20 = 2;
    if ( a4[33] == 1 )
    {
      v21 = *(_QWORD *)a4;
      v35 = *((_QWORD *)a4 + 1);
    }
    else
    {
      v21 = (__int64)a4;
      v19 = 2;
    }
    v46.m128i_i64[0] = (__int64)&v49;
    v47.m128i_i64[0] = v21;
    v46.m128i_i64[1] = v36;
    v47.m128i_i64[1] = v35;
    LOBYTE(v48) = 2;
    BYTE1(v48) = v19;
  }
  if ( BYTE1(v48) == 1 )
  {
    v6 = v46.m128i_i64[1];
    v22 = (__m128i *)v46.m128i_i64[0];
  }
  else
  {
    v22 = &v46;
    v20 = 2;
  }
  v43.m128i_i64[0] = (__int64)v22;
  v43.m128i_i64[1] = v6;
  v44.m128i_i8[0] = 45;
  LOBYTE(v45) = v20;
  BYTE1(v45) = 8;
  v23 = a5[32];
  if ( !v23 )
    goto LABEL_30;
  if ( v23 == 1 )
  {
    v40 = _mm_loadu_si128(&v43);
    v42 = v45;
    v41 = _mm_loadu_si128(&v44);
  }
  else
  {
    if ( a5[33] == 1 )
    {
      a6 = *((_QWORD *)a5 + 1);
      v24 = *(_QWORD *)a5;
    }
    else
    {
      v24 = (__int64)a5;
      v23 = 2;
    }
    v40.m128i_i64[0] = (__int64)&v43;
    v41.m128i_i64[0] = v24;
    v40.m128i_i64[1] = v34;
    v41.m128i_i64[1] = a6;
    LOBYTE(v42) = 2;
    BYTE1(v42) = v23;
  }
LABEL_31:
  sub_CA0F50((__int64 *)a1, (void **)&v40);
  sub_CA0F50((__int64 *)s1, (void **)a2);
  v25 = sub_CC8470((_DWORD *)s1[0], (unsigned __int64)s1[1]);
  v26 = (__m128i *)s1[0];
  *(_DWORD *)(a1 + 32) = v25;
  if ( v26 != &v56 )
    j_j___libc_free_0(v26, v56.m128i_i64[0] + 1);
  sub_CA0F50((__int64 *)s1, (void **)a2);
  v27 = sub_CC5470((_DWORD *)s1[0], (unsigned __int64)s1[1]);
  v28 = (__m128i *)s1[0];
  *(_DWORD *)(a1 + 36) = v27;
  if ( v28 != &v56 )
    j_j___libc_free_0(v28, v56.m128i_i64[0] + 1);
  sub_CA0F50((__int64 *)s1, (void **)a3);
  v29 = (__m128i *)s1[0];
  *(_DWORD *)(a1 + 40) = sub_CC4230((__int64)s1[0], (__int64)s1[1]);
  if ( v29 != &v56 )
    j_j___libc_free_0(v29, v56.m128i_i64[0] + 1);
  sub_CA0F50((__int64 *)s1, (void **)a4);
  v30 = (__m128i *)s1[0];
  *(_DWORD *)(a1 + 44) = sub_CC4400((__int64)s1[0], (unsigned __int64)s1[1]);
  if ( v30 != &v56 )
    j_j___libc_free_0(v30, v56.m128i_i64[0] + 1);
  sub_CA0F50((__int64 *)s1, (void **)a5);
  v31 = (__m128i *)s1[0];
  *(_DWORD *)(a1 + 48) = sub_CC4B20((__int64)s1[0], (unsigned __int64)s1[1]);
  if ( v31 != &v56 )
    j_j___libc_free_0(v31, v56.m128i_i64[0] + 1);
  sub_CA0F50((__int64 *)s1, (void **)a5);
  v32 = (__m128i *)s1[0];
  result = sub_CC4070((__int64)s1[0], (unsigned __int64)s1[1]);
  *(_DWORD *)(a1 + 52) = result;
  if ( v32 != &v56 )
  {
    j_j___libc_free_0(v32, v56.m128i_i64[0] + 1);
    result = *(unsigned int *)(a1 + 52);
  }
  if ( !(_DWORD)result )
  {
    result = sub_CC3FA0(a1);
    *(_DWORD *)(a1 + 52) = result;
  }
  return result;
}
