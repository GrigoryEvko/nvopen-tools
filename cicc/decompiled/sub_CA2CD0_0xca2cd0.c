// Function: sub_CA2CD0
// Address: 0xca2cd0
//
__int64 __fastcall sub_CA2CD0(__int64 a1, const __m128i *a2)
{
  bool v2; // zf
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  struct passwd *v5; // rsi
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // r14d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdx
  _QWORD *v17; // r9
  __int64 v18; // rdx
  char *v19; // rdi
  _QWORD *v20; // [rsp+10h] [rbp-4B0h]
  _QWORD *v21; // [rsp+10h] [rbp-4B0h]
  bool v22; // [rsp+3Fh] [rbp-481h] BYREF
  __m128i v23; // [rsp+40h] [rbp-480h] BYREF
  __int64 v24; // [rsp+50h] [rbp-470h]
  _BYTE v25[136]; // [rsp+58h] [rbp-468h] BYREF
  struct passwd v26[3]; // [rsp+E0h] [rbp-3E0h] BYREF
  __m128i v27; // [rsp+180h] [rbp-340h] BYREF
  __int64 v28; // [rsp+190h] [rbp-330h]
  _BYTE v29[136]; // [rsp+198h] [rbp-328h] BYREF
  char *v30; // [rsp+220h] [rbp-2A0h] BYREF
  __int64 v31; // [rsp+228h] [rbp-298h]
  __int64 v32; // [rsp+230h] [rbp-290h]
  _BYTE v33[128]; // [rsp+238h] [rbp-288h] BYREF
  char *v34; // [rsp+2B8h] [rbp-208h] BYREF
  __int64 v35; // [rsp+2C0h] [rbp-200h]
  __int64 v36; // [rsp+2C8h] [rbp-1F8h]
  _BYTE v37[128]; // [rsp+2D0h] [rbp-1F0h] BYREF
  __m128i v38; // [rsp+350h] [rbp-170h] BYREF
  __m128i v39; // [rsp+360h] [rbp-160h] BYREF
  __int64 v40; // [rsp+370h] [rbp-150h]
  char *v41; // [rsp+3E8h] [rbp-D8h] BYREF
  __int64 v42; // [rsp+3F0h] [rbp-D0h]
  __int64 v43; // [rsp+3F8h] [rbp-C8h]
  _BYTE v44[128]; // [rsp+400h] [rbp-C0h] BYREF
  char v45; // [rsp+480h] [rbp-40h]

  v2 = *(_BYTE *)(a1 + 328) == 0;
  v23.m128i_i64[0] = (__int64)v25;
  v23.m128i_i64[1] = 0;
  v24 = 128;
  v26[0].pw_name = (char *)&v26[0].pw_gecos;
  v26[0].pw_passwd = 0;
  *(_QWORD *)&v26[0].pw_uid = 128;
  v27.m128i_i64[0] = (__int64)v29;
  v27.m128i_i64[1] = 0;
  v28 = 128;
  if ( v2 || (*(_BYTE *)(a1 + 320) & 1) != 0 )
  {
    v3 = _mm_loadu_si128(a2);
    v4 = _mm_loadu_si128(a2 + 1);
    v40 = a2[2].m128i_i64[0];
    v38 = v3;
    v39 = v4;
  }
  else
  {
    sub_CA0EC0((__int64)a2, (__int64)&v27);
    v12 = *(_QWORD *)(a1 + 168);
    LOWORD(v40) = 261;
    v38.m128i_i64[0] = v12;
    v38.m128i_i64[1] = *(_QWORD *)(a1 + 176);
    sub_C846B0((__int64)&v38, (unsigned __int8 **)&v27);
    LOWORD(v40) = 261;
    v38 = v27;
  }
  sub_CA0EC0((__int64)&v38, (__int64)&v23);
  v5 = (struct passwd *)&v22;
  LOWORD(v40) = 261;
  v38 = v23;
  v6 = sub_C82790((__int64)&v38, &v22);
  if ( v6 )
    goto LABEL_7;
  if ( !v22 )
  {
    v10 = 20;
    sub_2241E50(&v38, &v22, v7, v8, v9);
    goto LABEL_8;
  }
  LOWORD(v40) = 261;
  v5 = v26;
  v38 = v23;
  v6 = sub_C84130((__int64)&v38, v26, 0, v8, v9);
  if ( v6 )
  {
LABEL_7:
    v10 = v6;
  }
  else
  {
    v31 = 0;
    v30 = v33;
    v32 = 128;
    if ( v23.m128i_i64[1] )
    {
      v5 = (struct passwd *)&v23;
      sub_CA1E30((__int64)&v30, (__int64)&v23, v13, v14, v15, (__int64)v26);
    }
    v35 = 0;
    v34 = v37;
    v36 = 128;
    if ( v26[0].pw_passwd )
    {
      v5 = v26;
      sub_CA1E30((__int64)&v34, (__int64)v26, v13, v14, v15, (__int64)v26);
    }
    v16 = v31;
    v17 = (_QWORD *)(a1 + 16);
    if ( *(_BYTE *)(a1 + 328) )
    {
      v45 &= ~1u;
      v38 = (__m128i)(unsigned __int64)&v39.m128i_u64[1];
      v39.m128i_i64[0] = 128;
      if ( v31 )
      {
        v5 = (struct passwd *)&v30;
        sub_CA1CD0((__int64)&v38, &v30, v31, v14, v15, (__int64)v17);
        v17 = (_QWORD *)(a1 + 16);
      }
      v18 = (__int64)v44;
      v42 = 0;
      v41 = v44;
      v43 = 128;
      if ( v35 )
      {
        v5 = (struct passwd *)&v34;
        v20 = v17;
        sub_CA1CD0((__int64)&v41, &v34, (__int64)v44, v14, v15, (__int64)v17);
        v18 = (__int64)v44;
        v17 = v20;
      }
      if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
      {
        v21 = v17;
        sub_CA1BE0(v17, (__int64)v5);
        v18 = (__int64)v44;
        v17 = v21;
      }
      v5 = (struct passwd *)v38.m128i_i64[1];
      if ( (v45 & 1) != 0 )
      {
        v18 = v38.m128i_u32[0];
        *(_BYTE *)(a1 + 320) |= 1u;
        *(_QWORD *)(a1 + 24) = v5;
        *(_DWORD *)(a1 + 16) = v18;
      }
      else
      {
        *(_BYTE *)(a1 + 320) &= ~1u;
        *(_QWORD *)(a1 + 16) = a1 + 40;
        *(_QWORD *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 32) = 128;
        if ( v5 )
        {
          sub_CA1CD0((__int64)v17, (char **)&v38, (__int64)v44, v14, v15, (__int64)v17);
          v18 = (__int64)v44;
        }
        v5 = (struct passwd *)(a1 + 192);
        v2 = v42 == 0;
        *(_QWORD *)(a1 + 176) = 0;
        *(_QWORD *)(a1 + 168) = a1 + 192;
        *(_QWORD *)(a1 + 184) = 128;
        if ( !v2 )
        {
          v5 = (struct passwd *)&v41;
          sub_CA1CD0(a1 + 168, &v41, (__int64)v44, v14, v15, (__int64)v17);
          v18 = (__int64)v44;
        }
        if ( (v45 & 1) == 0 )
        {
          if ( v41 != v44 )
            _libc_free(v41, v5);
          if ( (unsigned __int64 *)v38.m128i_i64[0] != &v39.m128i_u64[1] )
            _libc_free(v38.m128i_i64[0], v5);
        }
      }
    }
    else
    {
      v5 = (struct passwd *)(a1 + 40);
      *(_BYTE *)(a1 + 320) &= ~1u;
      *(_QWORD *)(a1 + 16) = a1 + 40;
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 128;
      if ( v16 )
      {
        v5 = (struct passwd *)&v30;
        sub_CA1CD0(a1 + 16, &v30, v16, v14, v15, (__int64)v17);
      }
      v18 = a1 + 192;
      v2 = v35 == 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = a1 + 192;
      *(_QWORD *)(a1 + 184) = 128;
      if ( !v2 )
      {
        v5 = (struct passwd *)&v34;
        sub_CA1CD0(a1 + 168, &v34, v18, v14, v15, (__int64)v17);
      }
      *(_BYTE *)(a1 + 328) = 1;
    }
    if ( v34 != v37 )
      _libc_free(v34, v5);
    v19 = v30;
    if ( v30 != v33 )
      _libc_free(v30, v5);
    v10 = 0;
    sub_2241E40(v19, v5, v18, v14, v15);
  }
LABEL_8:
  if ( (_BYTE *)v27.m128i_i64[0] != v29 )
    _libc_free(v27.m128i_i64[0], v5);
  if ( (char **)v26[0].pw_name != &v26[0].pw_gecos )
    _libc_free(v26[0].pw_name, v5);
  if ( (_BYTE *)v23.m128i_i64[0] != v25 )
    _libc_free(v23.m128i_i64[0], v5);
  return v10;
}
