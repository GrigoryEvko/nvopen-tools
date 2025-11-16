// Function: sub_1E926D0
// Address: 0x1e926d0
//
bool __fastcall sub_1E926D0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int32 v4; // r12d
  _QWORD *v5; // r13
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __m128i v11; // [rsp+20h] [rbp-300h] BYREF
  __int16 v12; // [rsp+30h] [rbp-2F0h]
  __m128i v13; // [rsp+40h] [rbp-2E0h] BYREF
  char v14; // [rsp+50h] [rbp-2D0h]
  char v15; // [rsp+51h] [rbp-2CFh]
  __m128i v16[2]; // [rsp+60h] [rbp-2C0h] BYREF
  __m128i v17; // [rsp+80h] [rbp-2A0h] BYREF
  char v18; // [rsp+90h] [rbp-290h]
  char v19; // [rsp+91h] [rbp-28Fh]
  __m128i v20[4]; // [rsp+A0h] [rbp-280h] BYREF
  __int64 v21; // [rsp+E8h] [rbp-238h]
  _BYTE *v22; // [rsp+F0h] [rbp-230h]
  _BYTE *v23; // [rsp+F8h] [rbp-228h]
  __int64 v24; // [rsp+100h] [rbp-220h]
  int v25; // [rsp+108h] [rbp-218h]
  _BYTE v26[64]; // [rsp+110h] [rbp-210h] BYREF
  unsigned __int64 v27; // [rsp+150h] [rbp-1D0h]
  __int64 v28; // [rsp+158h] [rbp-1C8h]
  int v29; // [rsp+160h] [rbp-1C0h]
  __int64 v30; // [rsp+168h] [rbp-1B8h]
  __int64 v31; // [rsp+170h] [rbp-1B0h]
  __int64 v32; // [rsp+178h] [rbp-1A8h]
  __int64 v33; // [rsp+180h] [rbp-1A0h]
  _BYTE *v34; // [rsp+188h] [rbp-198h]
  __int64 v35; // [rsp+190h] [rbp-190h]
  _BYTE v36[64]; // [rsp+198h] [rbp-188h] BYREF
  _BYTE *v37; // [rsp+1D8h] [rbp-148h]
  __int64 v38; // [rsp+1E0h] [rbp-140h]
  _BYTE v39[64]; // [rsp+1E8h] [rbp-138h] BYREF
  _BYTE *v40; // [rsp+228h] [rbp-F8h]
  __int64 v41; // [rsp+230h] [rbp-F0h]
  _BYTE v42[64]; // [rsp+238h] [rbp-E8h] BYREF
  _BYTE *v43; // [rsp+278h] [rbp-A8h]
  __int64 v44; // [rsp+280h] [rbp-A0h]
  _BYTE v45[32]; // [rsp+288h] [rbp-98h] BYREF
  __int64 v46; // [rsp+2A8h] [rbp-78h]
  __int64 v47; // [rsp+2B0h] [rbp-70h]
  _QWORD *v48; // [rsp+2B8h] [rbp-68h]
  __int64 v49; // [rsp+2C0h] [rbp-60h]
  unsigned int v50; // [rsp+2C8h] [rbp-58h]

  v22 = v26;
  v23 = v26;
  v34 = v36;
  v20[0].m128i_i64[0] = a2;
  v35 = 0x1000000000LL;
  v38 = 0x1000000000LL;
  v41 = 0x1000000000LL;
  v20[0].m128i_i64[1] = a3;
  v37 = v39;
  v21 = 0;
  v24 = 8;
  v25 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v40 = v42;
  v43 = v45;
  v44 = 0x400000000LL;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v4 = sub_1E8F660((__int64)v20, a1);
  if ( v50 )
  {
    v5 = v48;
    v6 = &v48[48 * v50];
    do
    {
      if ( *v5 != -8 && *v5 != -16 )
      {
        v7 = v5[37];
        if ( v7 != v5[36] )
          _libc_free(v7);
        v8 = v5[24];
        if ( v8 != v5[23] )
          _libc_free(v8);
        j___libc_free_0(v5[19]);
        j___libc_free_0(v5[15]);
        j___libc_free_0(v5[11]);
        j___libc_free_0(v5[7]);
        j___libc_free_0(v5[3]);
      }
      v5 += 48;
    }
    while ( v6 != v5 );
  }
  j___libc_free_0(v48);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  j___libc_free_0(v31);
  _libc_free(v27);
  if ( v23 != v22 )
    _libc_free((unsigned __int64)v23);
  if ( v4 && a4 )
  {
    v11.m128i_i32[0] = v4;
    v17.m128i_i64[0] = (__int64)" machine code errors.";
    v13.m128i_i64[0] = (__int64)"Found ";
    v19 = 1;
    v18 = 3;
    v12 = 265;
    v15 = 1;
    v14 = 3;
    sub_14EC200(v16, &v13, &v11);
    sub_14EC200(v20, v16, &v17);
    sub_16BCFB0((__int64)v20, 1u);
  }
  return v4 == 0;
}
