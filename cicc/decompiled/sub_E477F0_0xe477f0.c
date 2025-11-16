// Function: sub_E477F0
// Address: 0xe477f0
//
unsigned __int64 *__fastcall sub_E477F0(
        unsigned __int64 *a1,
        const char **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7)
{
  const char **v10; // rax
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // r15
  const char ***v15; // rsi
  _BYTE *v16; // r14
  _BYTE *v17; // r13
  _BYTE *v18; // rdi
  __int64 v20; // [rsp+20h] [rbp-210h]
  _QWORD v22[2]; // [rsp+30h] [rbp-200h] BYREF
  char v23; // [rsp+40h] [rbp-1F0h]
  _QWORD v24[2]; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v25; // [rsp+60h] [rbp-1D0h] BYREF
  __m128i *v26; // [rsp+70h] [rbp-1C0h]
  __int64 v27; // [rsp+78h] [rbp-1B8h]
  __m128i v28; // [rsp+80h] [rbp-1B0h] BYREF
  const char **v29; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v30; // [rsp+98h] [rbp-198h]
  _QWORD *v31; // [rsp+A0h] [rbp-190h] BYREF
  _QWORD v32[3]; // [rsp+B0h] [rbp-180h] BYREF
  int v33; // [rsp+C8h] [rbp-168h]
  _QWORD *v34; // [rsp+D0h] [rbp-160h] BYREF
  _QWORD v35[2]; // [rsp+E0h] [rbp-150h] BYREF
  _QWORD *v36; // [rsp+F0h] [rbp-140h]
  __int64 v37; // [rsp+F8h] [rbp-138h]
  _QWORD v38[2]; // [rsp+100h] [rbp-130h] BYREF
  __int64 v39; // [rsp+110h] [rbp-120h]
  __int64 v40; // [rsp+118h] [rbp-118h]
  __int64 v41; // [rsp+120h] [rbp-110h]
  _BYTE *v42; // [rsp+128h] [rbp-108h]
  __int64 v43; // [rsp+130h] [rbp-100h]
  _BYTE v44[248]; // [rsp+138h] [rbp-F8h] BYREF

  v29 = a2;
  v30 = a3;
  LOWORD(v32[0]) = 261;
  sub_C7EAD0((__int64)v22, &v29, 0, 1u, 0);
  if ( (v23 & 1) != 0 && LODWORD(v22[0]) )
  {
    (*(void (__fastcall **)(_QWORD *))(*(_QWORD *)v22[1] + 32LL))(v24);
    v12 = (__m128i *)sub_2241130(v24, 0, 0, "Could not open input file: ", 27);
    v26 = &v28;
    if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
    {
      v28 = _mm_loadu_si128(v12 + 1);
    }
    else
    {
      v26 = (__m128i *)v12->m128i_i64[0];
      v28.m128i_i64[0] = v12[1].m128i_i64[0];
    }
    v13 = v12->m128i_i64[1];
    v12[1].m128i_i8[0] = 0;
    v27 = v13;
    v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
    v14 = v26;
    v12->m128i_i64[1] = 0;
    v29 = 0;
    v20 = v27;
    v31 = v32;
    v30 = 0;
    sub_E45AE0((__int64 *)&v31, a2, (__int64)a2 + a3);
    v34 = v35;
    v32[2] = -1;
    v33 = 0;
    sub_E45AE0((__int64 *)&v34, v14, (__int64)v14->m128i_i64 + v20);
    v15 = &v29;
    v43 = 0x400000000LL;
    v36 = v38;
    v37 = 0;
    LOBYTE(v38[0]) = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = v44;
    sub_E45B90(a4, (__int64)&v29);
    v16 = v42;
    v17 = &v42[48 * (unsigned int)v43];
    if ( v42 != v17 )
    {
      do
      {
        v17 -= 48;
        v18 = (_BYTE *)*((_QWORD *)v17 + 2);
        if ( v18 != v17 + 32 )
        {
          v15 = (const char ***)(*((_QWORD *)v17 + 4) + 1LL);
          j_j___libc_free_0(v18, v15);
        }
      }
      while ( v16 != v17 );
      v17 = v42;
    }
    if ( v17 != v44 )
      _libc_free(v17, v15);
    if ( v39 )
      j_j___libc_free_0(v39, v41 - v39);
    if ( v36 != v38 )
      j_j___libc_free_0(v36, v38[0] + 1LL);
    if ( v34 != v35 )
      j_j___libc_free_0(v34, v35[0] + 1LL);
    if ( v31 != v32 )
      j_j___libc_free_0(v31, v32[0] + 1LL);
    if ( v26 != &v28 )
      j_j___libc_free_0(v26, v28.m128i_i64[0] + 1);
    if ( (__int64 *)v24[0] != &v25 )
      j_j___libc_free_0(v24[0], v25 + 1);
    *a1 = 0;
    if ( (v23 & 1) == 0 )
    {
LABEL_29:
      if ( v22[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v22[0] + 8LL))(v22[0]);
    }
  }
  else
  {
    v10 = (const char **)v22[0];
    v22[0] = 0;
    v29 = v10;
    sub_E472C0(a1, &v29, a4, a5, a6, a7);
    if ( v29 )
      (*((void (__fastcall **)(const char **))*v29 + 1))(v29);
    if ( (v23 & 1) == 0 )
      goto LABEL_29;
  }
  return a1;
}
