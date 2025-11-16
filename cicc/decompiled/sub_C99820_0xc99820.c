// Function: sub_C99820
// Address: 0xc99820
//
void __fastcall sub_C99820(__int64 a1)
{
  __int64 v2; // r13
  _OWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  _OWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // r13
  _OWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // r13
  size_t v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // r8
  size_t v22; // rdx
  __m128i *v23; // rdi
  __int64 v24; // r10
  size_t v25; // rdx
  void *v26; // [rsp+20h] [rbp-D0h] BYREF
  size_t v27; // [rsp+28h] [rbp-C8h]
  __m128i v28; // [rsp+30h] [rbp-C0h] BYREF
  void *dest; // [rsp+40h] [rbp-B0h] BYREF
  size_t v30; // [rsp+48h] [rbp-A8h]
  __m128i v31; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v32; // [rsp+60h] [rbp-90h] BYREF
  size_t n; // [rsp+68h] [rbp-88h]
  _QWORD src[4]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v35; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v36; // [rsp+98h] [rbp-58h]
  size_t v37; // [rsp+A0h] [rbp-50h]
  _OWORD v38[4]; // [rsp+A8h] [rbp-48h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = (_OWORD *)*(int *)(*(_QWORD *)(a1 + 8) + 16616LL);
  LOWORD(v35) = 3;
  v36 = v3;
  sub_C6B410(v2, (unsigned __int8 *)"pid", 3u);
  sub_C6C710(v2, (unsigned __int16 *)&v35, v4);
  sub_C6AE10(v2);
  sub_C6BC50((unsigned __int16 *)&v35);
  v6 = **(_OWORD ***)(a1 + 16);
  v5 = *(_QWORD *)a1;
  LOWORD(v35) = 3;
  v36 = v6;
  sub_C6B410(v5, (unsigned __int8 *)"tid", 3u);
  sub_C6C710(v5, (unsigned __int16 *)&v35, v7);
  sub_C6AE10(v5);
  sub_C6BC50((unsigned __int16 *)&v35);
  LOWORD(v32) = 5;
  v8 = *(_QWORD *)a1;
  n = (size_t)"X";
  src[0] = 1;
  if ( (unsigned __int8)sub_C6A630("X", 1, 0) )
    goto LABEL_2;
  sub_C6B0E0((__int64 *)&v26, (__int64)"X", 1u);
  LOWORD(v35) = 6;
  if ( !(unsigned __int8)sub_C6A630((char *)v26, v27, 0) )
  {
    sub_C6B0E0((__int64 *)&dest, (__int64)v26, v27);
    v23 = (__m128i *)v26;
    if ( dest == &v31 )
    {
      v25 = v30;
      if ( v30 )
      {
        if ( v30 == 1 )
          *(_BYTE *)v26 = v31.m128i_i8[0];
        else
          memcpy(v26, &v31, v30);
        v25 = v30;
        v23 = (__m128i *)v26;
      }
      v27 = v25;
      v23->m128i_i8[v25] = 0;
      v23 = (__m128i *)dest;
      goto LABEL_32;
    }
    if ( v26 == &v28 )
    {
      v26 = dest;
      v27 = v30;
      v28.m128i_i64[0] = v31.m128i_i64[0];
    }
    else
    {
      v24 = v28.m128i_i64[0];
      v26 = dest;
      v27 = v30;
      v28.m128i_i64[0] = v31.m128i_i64[0];
      if ( v23 )
      {
        dest = v23;
        v31.m128i_i64[0] = v24;
        goto LABEL_32;
      }
    }
    dest = &v31;
    v23 = &v31;
LABEL_32:
    v30 = 0;
    v23->m128i_i8[0] = 0;
    if ( dest != &v31 )
      j_j___libc_free_0(dest, v31.m128i_i64[0] + 1);
  }
  v36 = v38;
  if ( v26 == &v28 )
  {
    v38[0] = _mm_load_si128(&v28);
  }
  else
  {
    v36 = v26;
    *(_QWORD *)&v38[0] = v28.m128i_i64[0];
  }
  v26 = &v28;
  v37 = v27;
  v27 = 0;
  v28.m128i_i8[0] = 0;
  sub_C6BC50((unsigned __int16 *)&v32);
  sub_C6A4F0((__int64)&v32, (unsigned __int16 *)&v35);
  sub_C6BC50((unsigned __int16 *)&v35);
  if ( v26 != &v28 )
    j_j___libc_free_0(v26, v28.m128i_i64[0] + 1);
LABEL_2:
  sub_C6B410(v8, (unsigned __int8 *)"ph", 2u);
  sub_C6C710(v8, (unsigned __int16 *)&v32, v9);
  sub_C6AE10(v8);
  sub_C6BC50((unsigned __int16 *)&v32);
  v10 = *(_QWORD *)a1;
  LOWORD(v35) = 3;
  v36 = 0;
  sub_C6B410(v10, (unsigned __int8 *)"ts", 2u);
  sub_C6C710(v10, (unsigned __int16 *)&v35, v11);
  sub_C6AE10(v10);
  sub_C6BC50((unsigned __int16 *)&v35);
  v12 = *(_QWORD *)a1;
  v13 = **(_OWORD ***)(a1 + 24);
  v14 = *(_QWORD *)a1;
  LOWORD(v35) = 3;
  v36 = v13;
  sub_C6B410(v14, "dur", 3u);
  sub_C6C710(v12, (unsigned __int16 *)&v35, v15);
  sub_C6AE10(v12);
  sub_C6BC50((unsigned __int16 *)&v35);
  v16 = *(_QWORD *)a1;
  sub_8FD6D0((__int64)&dest, "Total ", *(_QWORD **)(a1 + 32));
  LOWORD(v35) = 6;
  if ( (unsigned __int8)sub_C6A630((char *)dest, v30, 0) )
    goto LABEL_3;
  sub_C6B0E0((__int64 *)&v32, (__int64)dest, v30);
  v20 = dest;
  if ( v32 == src )
  {
    v22 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v22 = n;
      v20 = dest;
    }
    v30 = v22;
    *((_BYTE *)v20 + v22) = 0;
    v20 = v32;
    goto LABEL_13;
  }
  if ( dest == &v31 )
  {
    dest = v32;
    v30 = n;
    v31.m128i_i64[0] = src[0];
  }
  else
  {
    v21 = v31.m128i_i64[0];
    dest = v32;
    v30 = n;
    v31.m128i_i64[0] = src[0];
    if ( v20 )
    {
      v32 = v20;
      src[0] = v21;
      goto LABEL_13;
    }
  }
  v32 = src;
  v20 = src;
LABEL_13:
  n = 0;
  *(_BYTE *)v20 = 0;
  if ( v32 != src )
    j_j___libc_free_0(v32, src[0] + 1LL);
LABEL_3:
  v36 = v38;
  if ( dest == &v31 )
  {
    v38[0] = _mm_load_si128(&v31);
  }
  else
  {
    v36 = dest;
    *(_QWORD *)&v38[0] = v31.m128i_i64[0];
  }
  dest = &v31;
  v37 = v30;
  v30 = 0;
  v31.m128i_i8[0] = 0;
  sub_C6B410(v16, (unsigned __int8 *)"name", 4u);
  sub_C6C710(v16, (unsigned __int16 *)&v35, v17);
  sub_C6AE10(v16);
  sub_C6BC50((unsigned __int16 *)&v35);
  if ( dest != &v31 )
    j_j___libc_free_0(dest, v31.m128i_i64[0] + 1);
  v18 = *(_QWORD *)a1;
  v36 = *(_OWORD **)(a1 + 40);
  v19 = *(_QWORD *)(a1 + 24);
  v35 = v18;
  v37 = v19;
  sub_C6B410(v18, (unsigned __int8 *)"args", 4u);
  sub_C6ACB0(v18);
  sub_C95E90((__int64)&v35);
  sub_C6AD90(v18);
  sub_C6AE10(v18);
}
