// Function: sub_243C570
// Address: 0x243c570
//
void __fastcall sub_243C570(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  unsigned __int64 *v12; // r13
  __int64 v13; // r8
  unsigned __int64 *v14; // r15
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  _BYTE *v21[2]; // [rsp+0h] [rbp-3E0h] BYREF
  __int64 v22; // [rsp+10h] [rbp-3D0h] BYREF
  __int64 *v23; // [rsp+20h] [rbp-3C0h]
  __int64 v24; // [rsp+28h] [rbp-3B8h]
  __int64 v25; // [rsp+30h] [rbp-3B0h] BYREF
  __m128i v26; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 *v27; // [rsp+50h] [rbp-390h] BYREF
  int v28; // [rsp+58h] [rbp-388h]
  char v29; // [rsp+5Ch] [rbp-384h]
  __int64 v30; // [rsp+60h] [rbp-380h] BYREF
  __m128i v31; // [rsp+68h] [rbp-378h] BYREF
  __int64 v32; // [rsp+78h] [rbp-368h]
  __m128i v33; // [rsp+80h] [rbp-360h] BYREF
  __m128i v34; // [rsp+90h] [rbp-350h]
  unsigned __int64 *v35; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v36; // [rsp+A8h] [rbp-338h]
  _BYTE v37[324]; // [rsp+B0h] [rbp-330h] BYREF
  int v38; // [rsp+1F4h] [rbp-1ECh]
  __int64 v39; // [rsp+1F8h] [rbp-1E8h]
  void *v40; // [rsp+200h] [rbp-1E0h] BYREF
  int v41; // [rsp+208h] [rbp-1D8h]
  char v42; // [rsp+20Ch] [rbp-1D4h]
  __int64 v43; // [rsp+210h] [rbp-1D0h]
  __m128i v44; // [rsp+218h] [rbp-1C8h] BYREF
  __int64 v45; // [rsp+228h] [rbp-1B8h]
  __m128i v46; // [rsp+230h] [rbp-1B0h] BYREF
  __m128i v47; // [rsp+240h] [rbp-1A0h] BYREF
  unsigned __int64 *v48; // [rsp+250h] [rbp-190h] BYREF
  unsigned int v49; // [rsp+258h] [rbp-188h]
  _BYTE v50[324]; // [rsp+260h] [rbp-180h] BYREF
  int v51; // [rsp+3A4h] [rbp-3Ch]
  __int64 v52; // [rsp+3A8h] [rbp-38h]

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v4)
    || (v19 = sub_B2BE50(v3),
        v20 = sub_B6F970(v19),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 48LL))(v20)) )
  {
    sub_B17770((__int64)&v40, (__int64)"hwasan", (__int64)"Sanitize", 8, (__int64)a2);
    sub_B18290((__int64)&v40, "Sanitized: F=", 0xDu);
    sub_B16080((__int64)v21, "Function", 8, a2);
    v27 = &v30;
    sub_2434550((__int64 *)&v27, v21[0], (__int64)&v21[0][(unsigned __int64)v21[1]]);
    v31.m128i_i64[1] = (__int64)&v33;
    sub_2434550(&v31.m128i_i64[1], v23, (__int64)v23 + v24);
    v34 = _mm_loadu_si128(&v26);
    sub_B180C0((__int64)&v40, (unsigned __int64)&v27);
    if ( (__m128i *)v31.m128i_i64[1] != &v33 )
      j_j___libc_free_0(v31.m128i_u64[1]);
    if ( v27 != &v30 )
      j_j___libc_free_0((unsigned __int64)v27);
    v9 = _mm_loadu_si128(&v44);
    v10 = _mm_loadu_si128(&v46);
    v11 = _mm_loadu_si128(&v47);
    v35 = (unsigned __int64 *)v37;
    v28 = v41;
    v31 = v9;
    v29 = v42;
    v33 = v10;
    v30 = v43;
    v34 = v11;
    v27 = (__int64 *)&unk_49D9D40;
    v32 = v45;
    v36 = 0x400000000LL;
    if ( v49 )
      sub_243C2F0((__int64)&v35, (__int64)&v48, v5, v6, v7, v8);
    v37[320] = v50[320];
    v38 = v51;
    v39 = v52;
    v27 = (__int64 *)&unk_49D9DB0;
    if ( v23 != &v25 )
      j_j___libc_free_0((unsigned __int64)v23);
    if ( (__int64 *)v21[0] != &v22 )
      j_j___libc_free_0((unsigned __int64)v21[0]);
    v12 = v48;
    v40 = &unk_49D9D40;
    v13 = 10LL * v49;
    v14 = &v48[v13];
    if ( v48 != &v48[v13] )
    {
      do
      {
        v14 -= 10;
        v15 = v14[4];
        if ( (unsigned __int64 *)v15 != v14 + 6 )
          j_j___libc_free_0(v15);
        if ( (unsigned __int64 *)*v14 != v14 + 2 )
          j_j___libc_free_0(*v14);
      }
      while ( v12 != v14 );
      v14 = v48;
    }
    if ( v14 != (unsigned __int64 *)v50 )
      _libc_free((unsigned __int64)v14);
    sub_1049740(a1, (__int64)&v27);
    v16 = v35;
    v27 = (__int64 *)&unk_49D9D40;
    v17 = &v35[10 * (unsigned int)v36];
    if ( v35 != v17 )
    {
      do
      {
        v17 -= 10;
        v18 = v17[4];
        if ( (unsigned __int64 *)v18 != v17 + 6 )
          j_j___libc_free_0(v18);
        if ( (unsigned __int64 *)*v17 != v17 + 2 )
          j_j___libc_free_0(*v17);
      }
      while ( v16 != v17 );
      v17 = v35;
    }
    if ( v17 != (unsigned __int64 *)v37 )
      _libc_free((unsigned __int64)v17);
  }
}
