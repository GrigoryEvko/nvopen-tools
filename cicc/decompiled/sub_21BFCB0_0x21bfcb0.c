// Function: sub_21BFCB0
// Address: 0x21bfcb0
//
__int64 __fastcall sub_21BFCB0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  const __m128i *v7; // r10
  __int64 v8; // rax
  _QWORD *v9; // r8
  unsigned int v10; // r9d
  unsigned __int16 v11; // ax
  __m128i v12; // xmm1
  int v13; // r14d
  const __m128i *v14; // r10
  __int64 v15; // r15
  _BYTE *v16; // rdx
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  unsigned __int8 v27; // si
  __int64 v28; // r14
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int8 v34; // si
  unsigned __int8 v35; // si
  __int128 v36; // [rsp-10h] [rbp-120h]
  __int64 v37; // [rsp+0h] [rbp-110h]
  const __m128i *v38; // [rsp+8h] [rbp-108h]
  _QWORD *v39; // [rsp+10h] [rbp-100h]
  __int64 v40; // [rsp+10h] [rbp-100h]
  __int64 v41; // [rsp+18h] [rbp-F8h]
  __m128i v42; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v43; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v44; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v46; // [rsp+50h] [rbp-C0h] BYREF
  int v47; // [rsp+58h] [rbp-B8h] BYREF
  unsigned __int8 v48; // [rsp+5Ch] [rbp-B4h]
  __int64 v49; // [rsp+60h] [rbp-B0h] BYREF
  int v50; // [rsp+68h] [rbp-A8h]
  _BYTE *v51; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v52; // [rsp+78h] [rbp-98h]
  _BYTE v53[144]; // [rsp+80h] [rbp-90h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v49 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v49, v6, 2);
  v7 = *(const __m128i **)(a2 + 32);
  v50 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(v7[2].m128i_i64[1] + 88);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = 0;
  v11 = *(_WORD *)(a2 + 24) - 675;
  if ( v11 <= 2u )
  {
    v12 = _mm_loadu_si128(v7);
    v13 = dword_433D960[v11];
    v51 = v53;
    v52 = 0x600000000LL;
    v42 = v12;
    if ( v13 )
    {
      v14 = v7 + 5;
      v15 = 120;
      v16 = v53;
      v17 = 40LL * (unsigned int)(v13 - 1) + 120;
      v18 = 0;
      while ( 1 )
      {
        a3 = _mm_loadu_si128(v14);
        *(__m128i *)&v16[16 * v18] = a3;
        v18 = (unsigned int)(v52 + 1);
        LODWORD(v52) = v52 + 1;
        if ( v15 == v17 )
          break;
        v14 = (const __m128i *)(v15 + *(_QWORD *)(a2 + 32));
        if ( HIDWORD(v52) <= (unsigned int)v18 )
        {
          v37 = v17;
          v38 = (const __m128i *)(v15 + *(_QWORD *)(a2 + 32));
          v39 = v9;
          sub_16CD150((__int64)&v51, v53, 0, 16, (int)v9, v17);
          v18 = (unsigned int)v52;
          v17 = v37;
          v14 = v38;
          v9 = v39;
        }
        v16 = v51;
        v15 += 40;
      }
    }
    v20 = sub_1D38BB0(
            *(_QWORD *)(a1 + 272),
            (unsigned int)v9,
            (__int64)&v49,
            5,
            0,
            1,
            a3,
            *(double *)v12.m128i_i64,
            a5,
            0);
    v21 = v19;
    v22 = (unsigned int)v52;
    if ( (unsigned int)v52 >= HIDWORD(v52) )
    {
      v40 = v20;
      v41 = v19;
      sub_16CD150((__int64)&v51, v53, 0, 16, v20, v19);
      v22 = (unsigned int)v52;
      v20 = v40;
      v21 = v41;
    }
    v23 = (__int64 *)&v51[16 * v22];
    *v23 = v20;
    v23[1] = v21;
    v24 = (unsigned int)(v52 + 1);
    LODWORD(v52) = v24;
    if ( HIDWORD(v52) <= (unsigned int)v24 )
    {
      sub_16CD150((__int64)&v51, v53, 0, 16, v20, v21);
      v24 = (unsigned int)v52;
    }
    *(__m128i *)&v51[16 * v24] = _mm_load_si128(&v42);
    LODWORD(v52) = v52 + 1;
    if ( v13 == 2 )
    {
      v35 = *(_BYTE *)(a2 + 88);
      v46 = 0x100001086LL;
      v45 = 0x100001084LL;
      v44 = 0x100001083LL;
      v43 = 0x100001089LL;
      sub_21BD570(
        (__int64)&v47,
        v35,
        4234,
        4231,
        4232,
        (__int64)&v43,
        (__int64)&v44,
        (__int64)&v45,
        4229,
        (__int64)&v46);
    }
    else if ( v13 == 4 )
    {
      v34 = *(_BYTE *)(a2 + 88);
      BYTE4(v46) = 0;
      v45 = 0x10000108CLL;
      BYTE4(v43) = 0;
      v44 = 0x10000108BLL;
      sub_21BD570(
        (__int64)&v47,
        v34,
        4240,
        4238,
        4239,
        (__int64)&v43,
        (__int64)&v44,
        (__int64)&v45,
        4237,
        (__int64)&v46);
    }
    else
    {
      v10 = 0;
      if ( v13 != 1 )
        goto LABEL_19;
      v27 = *(_BYTE *)(a2 + 88);
      v46 = 0x10000107ELL;
      v45 = 0x10000107CLL;
      v44 = 0x10000107BLL;
      v43 = 0x100001081LL;
      sub_21BD570(
        (__int64)&v47,
        v27,
        4226,
        4223,
        4224,
        (__int64)&v43,
        (__int64)&v44,
        (__int64)&v45,
        4221,
        (__int64)&v46);
    }
    v10 = v48;
    if ( !v48 )
    {
LABEL_19:
      v25 = (unsigned __int64)v51;
      if ( v51 == v53 )
        goto LABEL_21;
      goto LABEL_20;
    }
    *((_QWORD *)&v36 + 1) = (unsigned int)v52;
    *(_QWORD *)&v36 = v51;
    v28 = sub_1D2CDB0(*(_QWORD **)(a1 + 272), v47, (__int64)&v49, 1, 0, v48, v36);
    v29 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
    *v29 = *(_QWORD *)(a2 + 104);
    *(_QWORD *)(v28 + 88) = v29;
    *(_QWORD *)(v28 + 96) = v29 + 1;
    sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v28);
    sub_1D49010(v28);
    sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v30, v31, v32, v33);
    v25 = (unsigned __int64)v51;
    v10 = 1;
    if ( v51 != v53 )
    {
LABEL_20:
      v42.m128i_i8[0] = v10;
      _libc_free(v25);
      v10 = v42.m128i_u8[0];
    }
  }
LABEL_21:
  if ( v49 )
  {
    v42.m128i_i8[0] = v10;
    sub_161E7C0((__int64)&v49, v49);
    return v42.m128i_u8[0];
  }
  return v10;
}
