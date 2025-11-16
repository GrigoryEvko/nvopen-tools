// Function: sub_C97560
// Address: 0xc97560
//
void __fastcall sub_C97560(__int64 a1)
{
  __int64 v2; // r13
  _OWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  _OWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  _OWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r13
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // r13
  char *v22; // rsi
  unsigned __int64 v23; // rdx
  _OWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rdx
  _QWORD *v29; // rdi
  __int64 v30; // r9
  _QWORD *v31; // rdi
  __int64 v32; // r9
  size_t v33; // rdx
  size_t v34; // rdx
  _BYTE *v35; // rdi
  __m128i *v36; // rdx
  __int64 v37; // r9
  _BYTE *v38; // rdi
  __m128i *v39; // rdx
  __int64 v40; // r9
  _BYTE *v41; // rdi
  __int64 v42; // r9
  size_t v43; // rdx
  size_t v44; // rdx
  size_t v45; // rdx
  void *v46; // [rsp+20h] [rbp-D0h] BYREF
  size_t v47; // [rsp+28h] [rbp-C8h]
  __m128i v48; // [rsp+30h] [rbp-C0h] BYREF
  void *dest; // [rsp+40h] [rbp-B0h] BYREF
  size_t v50; // [rsp+48h] [rbp-A8h]
  __m128i v51; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v52; // [rsp+60h] [rbp-90h] BYREF
  size_t n; // [rsp+68h] [rbp-88h]
  _QWORD src[4]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v55; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v56; // [rsp+98h] [rbp-58h]
  size_t v57; // [rsp+A0h] [rbp-50h]
  _OWORD v58[4]; // [rsp+A8h] [rbp-48h] BYREF

  v3 = (_OWORD *)*(int *)(*(_QWORD *)(a1 + 8) + 16616LL);
  v2 = *(_QWORD *)a1;
  LOWORD(v55) = 3;
  v56 = v3;
  sub_C6B410(v2, (unsigned __int8 *)"pid", 3u);
  sub_C6C710(v2, (unsigned __int16 *)&v55, v4);
  sub_C6AE10(v2);
  sub_C6BC50((unsigned __int16 *)&v55);
  v6 = **(_OWORD ***)(a1 + 16);
  v5 = *(_QWORD *)a1;
  LOWORD(v55) = 3;
  v56 = v6;
  sub_C6B410(v5, (unsigned __int8 *)"tid", 3u);
  sub_C6C710(v5, (unsigned __int16 *)&v55, v7);
  sub_C6AE10(v5);
  sub_C6BC50((unsigned __int16 *)&v55);
  v9 = **(_OWORD ***)(a1 + 24);
  v8 = *(_QWORD *)a1;
  LOWORD(v55) = 3;
  v56 = v9;
  sub_C6B410(v8, (unsigned __int8 *)"ts", 2u);
  sub_C6C710(v8, (unsigned __int16 *)&v55, v10);
  sub_C6AE10(v8);
  sub_C6BC50((unsigned __int16 *)&v55);
  v11 = *(_QWORD *)(a1 + 32);
  v12 = *(_QWORD *)a1;
  v13 = *(_DWORD *)(v11 + 120);
  if ( v13 == 2 )
  {
    dest = &v51;
    sub_C95D30((__int64 *)&dest, *(_BYTE **)(v11 + 16), *(_QWORD *)(v11 + 16) + *(_QWORD *)(v11 + 24));
    LOWORD(v55) = 6;
    if ( (unsigned __int8)sub_C6A630((char *)dest, v50, 0) )
      goto LABEL_19;
    sub_C6B0E0((__int64 *)&v52, (__int64)dest, v50);
    v31 = dest;
    if ( v52 == src )
    {
      v34 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v34 = n;
        v31 = dest;
      }
      v50 = v34;
      *((_BYTE *)v31 + v34) = 0;
      v31 = v52;
      goto LABEL_38;
    }
    if ( dest == &v51 )
    {
      dest = v52;
      v50 = n;
      v51.m128i_i64[0] = src[0];
    }
    else
    {
      v32 = v51.m128i_i64[0];
      dest = v52;
      v50 = n;
      v51.m128i_i64[0] = src[0];
      if ( v31 )
      {
        v52 = v31;
        src[0] = v32;
        goto LABEL_38;
      }
    }
    v52 = src;
    v31 = src;
LABEL_38:
    n = 0;
    *(_BYTE *)v31 = 0;
    if ( v52 != src )
      j_j___libc_free_0(v52, src[0] + 1LL);
LABEL_19:
    v56 = v58;
    if ( dest == &v51 )
    {
      v58[0] = _mm_load_si128(&v51);
    }
    else
    {
      v56 = dest;
      *(_QWORD *)&v58[0] = v51.m128i_i64[0];
    }
    dest = &v51;
    v57 = v50;
    v50 = 0;
    v51.m128i_i8[0] = 0;
    sub_C6B410(v12, (unsigned __int8 *)"cat", 3u);
    sub_C6C710(v12, (unsigned __int16 *)&v55, v26);
    sub_C6AE10(v12);
    sub_C6BC50((unsigned __int16 *)&v55);
    if ( dest != &v51 )
      j_j___libc_free_0(dest, v51.m128i_i64[0] + 1);
    v27 = *(_QWORD *)a1;
    LOWORD(v52) = 5;
    n = (size_t)"b";
    src[0] = 1;
    if ( (unsigned __int8)sub_C6A630("b", 1, 0) )
      goto LABEL_24;
    sub_C6B0E0((__int64 *)&v46, (__int64)"b", 1u);
    LOWORD(v55) = 6;
    if ( (unsigned __int8)sub_C6A630((char *)v46, v47, 0) )
    {
LABEL_56:
      v56 = v58;
      if ( v46 == &v48 )
      {
        v58[0] = _mm_load_si128(&v48);
      }
      else
      {
        v56 = v46;
        *(_QWORD *)&v58[0] = v48.m128i_i64[0];
      }
      v46 = &v48;
      v57 = v47;
      v47 = 0;
      v48.m128i_i8[0] = 0;
      sub_C6BC50((unsigned __int16 *)&v52);
      sub_C6A4F0((__int64)&v52, (unsigned __int16 *)&v55);
      sub_C6BC50((unsigned __int16 *)&v55);
      if ( v46 != &v48 )
        j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
LABEL_24:
      sub_C6B410(v27, (unsigned __int8 *)"ph", 2u);
      sub_C6C710(v27, (unsigned __int16 *)&v52, v28);
      sub_C6AE10(v27);
      sub_C6BC50((unsigned __int16 *)&v52);
      v21 = *(_QWORD *)a1;
      v56 = 0;
      LOWORD(v55) = 3;
      v22 = "id";
      v23 = 2;
      goto LABEL_16;
    }
    sub_C6B0E0((__int64 *)&dest, (__int64)v46, v47);
    v41 = v46;
    if ( dest == &v51 )
    {
      v45 = v50;
      if ( v50 )
      {
        if ( v50 == 1 )
          *(_BYTE *)v46 = v51.m128i_i8[0];
        else
          memcpy(v46, &v51, v50);
        v45 = v50;
        v41 = v46;
      }
      v47 = v45;
      v41[v45] = 0;
      goto LABEL_87;
    }
    if ( v46 == &v48 )
    {
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
    }
    else
    {
      v42 = v48.m128i_i64[0];
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
      if ( v41 )
      {
        dest = v41;
        v51.m128i_i64[0] = v42;
        goto LABEL_87;
      }
    }
    dest = &v51;
LABEL_87:
    v50 = 0;
    *(_BYTE *)dest = 0;
    if ( dest != &v51 )
      j_j___libc_free_0(dest, v51.m128i_i64[0] + 1);
    goto LABEL_56;
  }
  LOWORD(v52) = 5;
  if ( v13 )
  {
    src[0] = 1;
    n = (size_t)"i";
    if ( (unsigned __int8)sub_C6A630("i", 1, 0) )
    {
LABEL_4:
      sub_C6B410(v12, (unsigned __int8 *)"ph", 2u);
      sub_C6C710(v12, (unsigned __int16 *)&v52, v14);
      sub_C6AE10(v12);
      sub_C6BC50((unsigned __int16 *)&v52);
      goto LABEL_5;
    }
    sub_C6B0E0((__int64 *)&v46, (__int64)"i", 1u);
    LOWORD(v55) = 6;
    if ( (unsigned __int8)sub_C6A630((char *)v46, v47, 0) )
    {
LABEL_41:
      v56 = v58;
      if ( v46 == &v48 )
      {
        v58[0] = _mm_load_si128(&v48);
      }
      else
      {
        v56 = v46;
        *(_QWORD *)&v58[0] = v48.m128i_i64[0];
      }
      v46 = &v48;
      v57 = v47;
      v47 = 0;
      v48.m128i_i8[0] = 0;
      sub_C6BC50((unsigned __int16 *)&v52);
      sub_C6A4F0((__int64)&v52, (unsigned __int16 *)&v55);
      sub_C6BC50((unsigned __int16 *)&v55);
      if ( v46 != &v48 )
        j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
      goto LABEL_4;
    }
    sub_C6B0E0((__int64 *)&dest, (__int64)v46, v47);
    v38 = v46;
    v39 = (__m128i *)v46;
    if ( dest == &v51 )
    {
      v44 = v50;
      if ( v50 )
      {
        if ( v50 == 1 )
          *(_BYTE *)v46 = v51.m128i_i8[0];
        else
          memcpy(v46, &v51, v50);
        v44 = v50;
        v38 = v46;
      }
      v47 = v44;
      v38[v44] = 0;
      v39 = (__m128i *)dest;
      goto LABEL_81;
    }
    if ( v46 == &v48 )
    {
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
    }
    else
    {
      v40 = v48.m128i_i64[0];
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
      if ( v39 )
      {
        dest = v39;
        v51.m128i_i64[0] = v40;
        goto LABEL_81;
      }
    }
    dest = &v51;
    v39 = &v51;
LABEL_81:
    v50 = 0;
    v39->m128i_i8[0] = 0;
    if ( dest != &v51 )
      j_j___libc_free_0(dest, v51.m128i_i64[0] + 1);
    goto LABEL_41;
  }
  src[0] = 1;
  n = (size_t)"X";
  if ( (unsigned __int8)sub_C6A630("X", 1, 0) )
    goto LABEL_15;
  sub_C6B0E0((__int64 *)&v46, (__int64)"X", 1u);
  LOWORD(v55) = 6;
  if ( !(unsigned __int8)sub_C6A630((char *)v46, v47, 0) )
  {
    sub_C6B0E0((__int64 *)&dest, (__int64)v46, v47);
    v35 = v46;
    v36 = (__m128i *)v46;
    if ( dest == &v51 )
    {
      v43 = v50;
      if ( v50 )
      {
        if ( v50 == 1 )
          *(_BYTE *)v46 = v51.m128i_i8[0];
        else
          memcpy(v46, &v51, v50);
        v43 = v50;
        v35 = v46;
      }
      v47 = v43;
      v35[v43] = 0;
      v36 = (__m128i *)dest;
      goto LABEL_75;
    }
    if ( v46 == &v48 )
    {
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
    }
    else
    {
      v37 = v48.m128i_i64[0];
      v46 = dest;
      v47 = v50;
      v48.m128i_i64[0] = v51.m128i_i64[0];
      if ( v36 )
      {
        dest = v36;
        v51.m128i_i64[0] = v37;
        goto LABEL_75;
      }
    }
    dest = &v51;
    v36 = &v51;
LABEL_75:
    v50 = 0;
    v36->m128i_i8[0] = 0;
    if ( dest != &v51 )
      j_j___libc_free_0(dest, v51.m128i_i64[0] + 1);
  }
  v56 = v58;
  if ( v46 == &v48 )
  {
    v58[0] = _mm_load_si128(&v48);
  }
  else
  {
    v56 = v46;
    *(_QWORD *)&v58[0] = v48.m128i_i64[0];
  }
  v46 = &v48;
  v57 = v47;
  v47 = 0;
  v48.m128i_i8[0] = 0;
  sub_C6BC50((unsigned __int16 *)&v52);
  sub_C6A4F0((__int64)&v52, (unsigned __int16 *)&v55);
  sub_C6BC50((unsigned __int16 *)&v55);
  if ( v46 != &v48 )
    j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
LABEL_15:
  sub_C6B410(v12, (unsigned __int8 *)"ph", 2u);
  sub_C6C710(v12, (unsigned __int16 *)&v52, v20);
  sub_C6AE10(v12);
  sub_C6BC50((unsigned __int16 *)&v52);
  v21 = *(_QWORD *)a1;
  v22 = "dur";
  v23 = 3;
  v24 = **(_OWORD ***)(a1 + 40);
  LOWORD(v55) = 3;
  v56 = v24;
LABEL_16:
  sub_C6B410(v21, (unsigned __int8 *)v22, v23);
  sub_C6C710(v21, (unsigned __int16 *)&v55, v25);
  sub_C6AE10(v21);
  sub_C6BC50((unsigned __int16 *)&v55);
LABEL_5:
  v15 = *(_QWORD *)(a1 + 32);
  v16 = *(_QWORD *)a1;
  dest = &v51;
  sub_C95D30((__int64 *)&dest, *(_BYTE **)(v15 + 16), *(_QWORD *)(v15 + 16) + *(_QWORD *)(v15 + 24));
  LOWORD(v55) = 6;
  if ( (unsigned __int8)sub_C6A630((char *)dest, v50, 0) )
    goto LABEL_6;
  sub_C6B0E0((__int64 *)&v52, (__int64)dest, v50);
  v29 = dest;
  if ( v52 == src )
  {
    v33 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v33 = n;
      v29 = dest;
    }
    v50 = v33;
    *((_BYTE *)v29 + v33) = 0;
    v29 = v52;
    goto LABEL_30;
  }
  if ( dest == &v51 )
  {
    dest = v52;
    v50 = n;
    v51.m128i_i64[0] = src[0];
  }
  else
  {
    v30 = v51.m128i_i64[0];
    dest = v52;
    v50 = n;
    v51.m128i_i64[0] = src[0];
    if ( v29 )
    {
      v52 = v29;
      src[0] = v30;
      goto LABEL_30;
    }
  }
  v52 = src;
  v29 = src;
LABEL_30:
  n = 0;
  *(_BYTE *)v29 = 0;
  if ( v52 != src )
    j_j___libc_free_0(v52, src[0] + 1LL);
LABEL_6:
  v56 = v58;
  if ( dest == &v51 )
  {
    v58[0] = _mm_load_si128(&v51);
  }
  else
  {
    v56 = dest;
    *(_QWORD *)&v58[0] = v51.m128i_i64[0];
  }
  dest = &v51;
  v57 = v50;
  v50 = 0;
  v51.m128i_i8[0] = 0;
  sub_C6B410(v16, (unsigned __int8 *)"name", 4u);
  sub_C6C710(v16, (unsigned __int16 *)&v55, v17);
  sub_C6AE10(v16);
  sub_C6BC50((unsigned __int16 *)&v55);
  if ( dest != &v51 )
    j_j___libc_free_0(dest, v51.m128i_i64[0] + 1);
  v18 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(v18 + 56) || *(_QWORD *)(v18 + 88) )
  {
    v19 = *(_QWORD *)a1;
    v55 = *(_QWORD *)(a1 + 32);
    v56 = (_OWORD *)v19;
    sub_C6B410(v19, (unsigned __int8 *)"args", 4u);
    sub_C6ACB0(v19);
    sub_C96AA0(&v55);
    sub_C6AD90(v19);
    sub_C6AE10(v19);
  }
}
