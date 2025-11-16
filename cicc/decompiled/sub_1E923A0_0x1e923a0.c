// Function: sub_1E923A0
// Address: 0x1e923a0
//
__int64 __fastcall sub_1E923A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int32 v8; // [rsp+1Ch] [rbp-304h]
  __m128i v9; // [rsp+20h] [rbp-300h] BYREF
  __int16 v10; // [rsp+30h] [rbp-2F0h]
  __m128i v11; // [rsp+40h] [rbp-2E0h] BYREF
  char v12; // [rsp+50h] [rbp-2D0h]
  char v13; // [rsp+51h] [rbp-2CFh]
  __m128i v14[2]; // [rsp+60h] [rbp-2C0h] BYREF
  __m128i v15; // [rsp+80h] [rbp-2A0h] BYREF
  char v16; // [rsp+90h] [rbp-290h]
  char v17; // [rsp+91h] [rbp-28Fh]
  __m128i v18[4]; // [rsp+A0h] [rbp-280h] BYREF
  __int64 v19; // [rsp+E8h] [rbp-238h]
  _BYTE *v20; // [rsp+F0h] [rbp-230h]
  _BYTE *v21; // [rsp+F8h] [rbp-228h]
  __int64 v22; // [rsp+100h] [rbp-220h]
  int v23; // [rsp+108h] [rbp-218h]
  _BYTE v24[64]; // [rsp+110h] [rbp-210h] BYREF
  unsigned __int64 v25; // [rsp+150h] [rbp-1D0h]
  __int64 v26; // [rsp+158h] [rbp-1C8h]
  int v27; // [rsp+160h] [rbp-1C0h]
  __int64 v28; // [rsp+168h] [rbp-1B8h]
  __int64 v29; // [rsp+170h] [rbp-1B0h]
  __int64 v30; // [rsp+178h] [rbp-1A8h]
  __int64 v31; // [rsp+180h] [rbp-1A0h]
  _BYTE *v32; // [rsp+188h] [rbp-198h]
  __int64 v33; // [rsp+190h] [rbp-190h]
  _BYTE v34[64]; // [rsp+198h] [rbp-188h] BYREF
  _BYTE *v35; // [rsp+1D8h] [rbp-148h]
  __int64 v36; // [rsp+1E0h] [rbp-140h]
  _BYTE v37[64]; // [rsp+1E8h] [rbp-138h] BYREF
  _BYTE *v38; // [rsp+228h] [rbp-F8h]
  __int64 v39; // [rsp+230h] [rbp-F0h]
  _BYTE v40[64]; // [rsp+238h] [rbp-E8h] BYREF
  _BYTE *v41; // [rsp+278h] [rbp-A8h]
  __int64 v42; // [rsp+280h] [rbp-A0h]
  _BYTE v43[32]; // [rsp+288h] [rbp-98h] BYREF
  __int64 v44; // [rsp+2A8h] [rbp-78h]
  __int64 v45; // [rsp+2B0h] [rbp-70h]
  _QWORD *v46; // [rsp+2B8h] [rbp-68h]
  __int64 v47; // [rsp+2C0h] [rbp-60h]
  unsigned int v48; // [rsp+2C8h] [rbp-58h]

  v2 = *(_QWORD *)(a1 + 232);
  v18[0].m128i_i64[0] = a1;
  v18[0].m128i_i64[1] = v2;
  v20 = v24;
  v21 = v24;
  v32 = v34;
  v33 = 0x1000000000LL;
  v36 = 0x1000000000LL;
  v39 = 0x1000000000LL;
  v42 = 0x400000000LL;
  v19 = 0;
  v22 = 8;
  v23 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v35 = v37;
  v38 = v40;
  v41 = v43;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v8 = sub_1E8F660((__int64)v18, a2);
  if ( v48 )
  {
    v3 = v46;
    v4 = &v46[48 * v48];
    do
    {
      if ( *v3 != -8 && *v3 != -16 )
      {
        v5 = v3[37];
        if ( v5 != v3[36] )
          _libc_free(v5);
        v6 = v3[24];
        if ( v6 != v3[23] )
          _libc_free(v6);
        j___libc_free_0(v3[19]);
        j___libc_free_0(v3[15]);
        j___libc_free_0(v3[11]);
        j___libc_free_0(v3[7]);
        j___libc_free_0(v3[3]);
      }
      v3 += 48;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(v46);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  j___libc_free_0(v29);
  _libc_free(v25);
  if ( v21 != v20 )
    _libc_free((unsigned __int64)v21);
  if ( v8 )
  {
    v17 = 1;
    v15.m128i_i64[0] = (__int64)" machine code errors.";
    v16 = 3;
    v9.m128i_i32[0] = v8;
    v11.m128i_i64[0] = (__int64)"Found ";
    v10 = 265;
    v13 = 1;
    v12 = 3;
    sub_14EC200(v14, &v11, &v9);
    sub_14EC200(v18, v14, &v15);
    sub_16BCFB0((__int64)v18, 1u);
  }
  return 0;
}
