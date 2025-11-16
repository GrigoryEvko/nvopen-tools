// Function: sub_ACF990
// Address: 0xacf990
//
__int64 __fastcall sub_ACF990(__int64 a1)
{
  __int64 v1; // rcx
  unsigned __int8 v2; // dl
  __int16 v3; // ax
  __int64 v4; // rdx
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // r14
  __int64 i; // r12
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __m128i v14; // xmm1
  __m128i v15; // xmm0
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // r12d
  __int64 v20; // [rsp+8h] [rbp-218h]
  unsigned __int64 v21; // [rsp+10h] [rbp-210h] BYREF
  unsigned __int64 v22; // [rsp+18h] [rbp-208h] BYREF
  __int16 v23; // [rsp+20h] [rbp-200h]
  __m128i v24; // [rsp+28h] [rbp-1F8h] BYREF
  __m128i v25; // [rsp+38h] [rbp-1E8h] BYREF
  __int64 v26; // [rsp+48h] [rbp-1D8h]
  __int64 v27; // [rsp+50h] [rbp-1D0h] BYREF
  unsigned int v28; // [rsp+58h] [rbp-1C8h]
  __int64 v29; // [rsp+60h] [rbp-1C0h]
  unsigned int v30; // [rsp+68h] [rbp-1B8h]
  char v31; // [rsp+70h] [rbp-1B0h]
  __int64 v32; // [rsp+80h] [rbp-1A0h] BYREF
  char v33[8]; // [rsp+88h] [rbp-198h] BYREF
  __m128i v34; // [rsp+90h] [rbp-190h]
  __m128i v35; // [rsp+A0h] [rbp-180h]
  __int64 v36; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v37; // [rsp+B8h] [rbp-168h]
  unsigned int v38; // [rsp+C0h] [rbp-160h]
  __int64 v39; // [rsp+C8h] [rbp-158h]
  unsigned int v40; // [rsp+D0h] [rbp-150h]
  char v41; // [rsp+D8h] [rbp-148h]
  _BYTE *v42; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v43; // [rsp+E8h] [rbp-138h]
  _BYTE v44[304]; // [rsp+F0h] [rbp-130h] BYREF

  v1 = 0;
  v2 = *(_BYTE *)(a1 + 1);
  v43 = 0x2000000000LL;
  v3 = *(_WORD *)(a1 + 2);
  v42 = v44;
  v24 = 0u;
  HIBYTE(v23) = v2 >> 1;
  v4 = 0;
  LOBYTE(v23) = v3;
  if ( v3 == 63 )
  {
    v1 = sub_AC35F0(a1);
    v3 = *(_WORD *)(a1 + 2);
  }
  v25.m128i_i64[0] = v1;
  v25.m128i_i64[1] = v4;
  if ( v3 == 34 )
  {
    v5 = sub_AC5180(a1);
    v6 = *(_WORD *)(a1 + 2) == 34;
    v26 = v5;
    if ( v6 )
    {
      sub_AC51A0((__int64)&v27, a1);
      goto LABEL_6;
    }
  }
  else
  {
    v26 = 0;
  }
  v31 = 0;
LABEL_6:
  v7 = (unsigned int)v43;
  v8 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v8 )
  {
    v9 = v8 - 1;
    for ( i = 0; ; ++i )
    {
      v11 = i - v8;
      v12 = *(_QWORD *)(a1 + 32 * v11);
      if ( v7 + 1 > (unsigned __int64)HIDWORD(v43) )
      {
        v20 = *(_QWORD *)(a1 + 32 * v11);
        sub_C8D5F0(&v42, v44, v7 + 1, 8);
        v7 = (unsigned int)v43;
        v12 = v20;
      }
      *(_QWORD *)&v42[8 * v7] = v12;
      v7 = (unsigned int)(v43 + 1);
      LODWORD(v43) = v43 + 1;
      if ( i == v9 )
        break;
      v8 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    }
  }
  v24.m128i_i64[1] = v7;
  v13 = *(_QWORD *)(a1 + 8);
  v14 = _mm_loadu_si128(&v25);
  v41 = 0;
  v32 = v13;
  v24.m128i_i64[0] = (__int64)v42;
  v15 = _mm_loadu_si128(&v24);
  *(_WORD *)v33 = v23;
  v34 = v15;
  v36 = v26;
  v35 = v14;
  if ( v31 )
  {
    v16 = v28;
    v41 = 1;
    v28 = 0;
    v38 = v16;
    v37 = v27;
    v17 = v30;
    v30 = 0;
    v40 = v17;
    v39 = v29;
  }
  v22 = sub_AC61D0((__int64 *)v35.m128i_i64[0], v35.m128i_i64[0] + 4 * v35.m128i_i64[1]);
  v21 = sub_AC5F60((__int64 *)v34.m128i_i64[0], v34.m128i_i64[0] + 8 * v34.m128i_i64[1]);
  LODWORD(v22) = sub_AC5EC0(v33, &v33[1], (__int64 *)&v21, (__int64 *)&v22, &v36);
  v18 = sub_AC7AE0(&v32, &v22);
  if ( v41 )
  {
    v41 = 0;
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
  }
  if ( v31 )
  {
    v31 = 0;
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
  }
  if ( v42 != v44 )
    _libc_free(v42, &v22);
  return v18;
}
