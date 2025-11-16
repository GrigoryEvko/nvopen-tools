// Function: sub_310D050
// Address: 0x310d050
//
void __fastcall sub_310D050(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // r13
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r13
  _OWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r13
  _QWORD *v14; // rdi
  __int64 v15; // r8
  size_t v16; // rdx
  __m128i *v17; // rdi
  __int64 v18; // r9
  size_t v19; // rdx
  __int64 v20; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v21; // [rsp+18h] [rbp-D8h]
  void *v22; // [rsp+20h] [rbp-D0h] BYREF
  size_t v23; // [rsp+28h] [rbp-C8h]
  __m128i v24; // [rsp+30h] [rbp-C0h] BYREF
  void *dest; // [rsp+40h] [rbp-B0h] BYREF
  size_t v26; // [rsp+48h] [rbp-A8h]
  __m128i v27; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v28; // [rsp+60h] [rbp-90h] BYREF
  size_t n; // [rsp+68h] [rbp-88h]
  _QWORD src[4]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v31; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v32; // [rsp+98h] [rbp-58h]
  size_t v33; // [rsp+A0h] [rbp-50h]
  _OWORD v34[4]; // [rsp+A8h] [rbp-48h] BYREF

  v2 = a1[1];
  v3 = *a1;
  dest = &v27;
  sub_11F4570((__int64 *)&dest, *(_BYTE **)v2, *(_QWORD *)v2 + *(_QWORD *)(v2 + 8));
  LOWORD(v31) = 6;
  if ( (unsigned __int8)sub_C6A630((char *)dest, v26, 0) )
    goto LABEL_2;
  sub_C6B0E0((__int64 *)&v28, (__int64)dest, v26);
  v14 = dest;
  if ( v28 == src )
  {
    v16 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v16 = n;
      v14 = dest;
    }
    v26 = v16;
    *((_BYTE *)v14 + v16) = 0;
    v14 = v28;
    goto LABEL_13;
  }
  if ( dest == &v27 )
  {
    dest = v28;
    v26 = n;
    v27.m128i_i64[0] = src[0];
  }
  else
  {
    v15 = v27.m128i_i64[0];
    dest = v28;
    v26 = n;
    v27.m128i_i64[0] = src[0];
    if ( v14 )
    {
      v28 = v14;
      src[0] = v15;
      goto LABEL_13;
    }
  }
  v28 = src;
  v14 = src;
LABEL_13:
  n = 0;
  *(_BYTE *)v14 = 0;
  if ( v28 != src )
    j_j___libc_free_0((unsigned __int64)v28);
LABEL_2:
  v32 = v34;
  if ( dest == &v27 )
  {
    v34[0] = _mm_load_si128(&v27);
  }
  else
  {
    v32 = dest;
    *(_QWORD *)&v34[0] = v27.m128i_i64[0];
  }
  dest = &v27;
  v33 = v26;
  v26 = 0;
  v27.m128i_i8[0] = 0;
  sub_C6B410(v3, (unsigned __int8 *)"name", 4u);
  sub_C6C710(v3, (unsigned __int16 *)&v31, v4);
  sub_C6AE10(v3);
  sub_C6BC50((unsigned __int16 *)&v31);
  if ( dest != &v27 )
    j_j___libc_free_0((unsigned __int64)dest);
  v5 = *a1;
  v6 = (char *)sub_310D030(*(_DWORD *)(a1[1] + 36));
  LOWORD(v28) = 5;
  src[0] = v7;
  n = (size_t)v6;
  v20 = (__int64)v6;
  v21 = v7;
  if ( !(unsigned __int8)sub_C6A630(v6, v7, 0) )
  {
    sub_C6B0E0((__int64 *)&v22, v20, v21);
    LOWORD(v31) = 6;
    if ( (unsigned __int8)sub_C6A630((char *)v22, v23, 0) )
    {
LABEL_16:
      v32 = v34;
      if ( v22 == &v24 )
      {
        v34[0] = _mm_load_si128(&v24);
      }
      else
      {
        v32 = v22;
        *(_QWORD *)&v34[0] = v24.m128i_i64[0];
      }
      v22 = &v24;
      v33 = v23;
      v23 = 0;
      v24.m128i_i8[0] = 0;
      sub_C6BC50((unsigned __int16 *)&v28);
      sub_C6A4F0((__int64)&v28, (unsigned __int16 *)&v31);
      sub_C6BC50((unsigned __int16 *)&v31);
      if ( v22 != &v24 )
        j_j___libc_free_0((unsigned __int64)v22);
      goto LABEL_7;
    }
    sub_C6B0E0((__int64 *)&dest, (__int64)v22, v23);
    v17 = (__m128i *)v22;
    if ( dest == &v27 )
    {
      v19 = v26;
      if ( v26 )
      {
        if ( v26 == 1 )
          *(_BYTE *)v22 = v27.m128i_i8[0];
        else
          memcpy(v22, &v27, v26);
        v19 = v26;
        v17 = (__m128i *)v22;
      }
      v23 = v19;
      v17->m128i_i8[v19] = 0;
      v17 = (__m128i *)dest;
      goto LABEL_32;
    }
    if ( v22 == &v24 )
    {
      v22 = dest;
      v23 = v26;
      v24.m128i_i64[0] = v27.m128i_i64[0];
    }
    else
    {
      v18 = v24.m128i_i64[0];
      v22 = dest;
      v23 = v26;
      v24.m128i_i64[0] = v27.m128i_i64[0];
      if ( v17 )
      {
        dest = v17;
        v27.m128i_i64[0] = v18;
        goto LABEL_32;
      }
    }
    dest = &v27;
    v17 = &v27;
LABEL_32:
    v26 = 0;
    v17->m128i_i8[0] = 0;
    if ( dest != &v27 )
      j_j___libc_free_0((unsigned __int64)dest);
    goto LABEL_16;
  }
LABEL_7:
  sub_C6B410(v5, (unsigned __int8 *)"type", 4u);
  sub_C6C710(v5, (unsigned __int16 *)&v28, v8);
  sub_C6AE10(v5);
  sub_C6BC50((unsigned __int16 *)&v28);
  v9 = *a1;
  v10 = (_OWORD *)*(int *)(a1[1] + 32);
  v11 = *a1;
  LOWORD(v31) = 3;
  v32 = v10;
  sub_C6B410(v11, (unsigned __int8 *)"port", 4u);
  sub_C6C710(v9, (unsigned __int16 *)&v31, v12);
  sub_C6AE10(v9);
  sub_C6BC50((unsigned __int16 *)&v31);
  v13 = *a1;
  v31 = a1[1];
  v32 = (_OWORD *)v13;
  sub_C6B410(v13, (unsigned __int8 *)"shape", 5u);
  sub_C6AB50(v13);
  sub_310CF40(&v31);
  sub_C6AC30(v13);
  sub_C6AE10(v13);
}
