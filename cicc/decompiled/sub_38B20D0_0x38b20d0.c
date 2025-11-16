// Function: sub_38B20D0
// Address: 0x38b20d0
//
__int64 __fastcall sub_38B20D0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v12; // r14
  bool v13; // zf
  __int64 *v14; // r12
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r15
  int v17; // eax
  int v18; // eax
  __int64 v19; // rdi
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned __int64 v22; // rsi
  unsigned int v23; // r15d
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int64 v26; // rsi
  __m128i *v28; // rax
  __int64 v29; // [rsp+0h] [rbp-E0h]
  __int64 *v30; // [rsp+8h] [rbp-D8h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  unsigned int v32; // [rsp+2Ch] [rbp-B4h]
  _OWORD **v33; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v34; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v35[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v36[16]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v37[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v38; // [rsp+80h] [rbp-60h] BYREF
  _OWORD *v39; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 v40; // [rsp+98h] [rbp-48h]
  _OWORD v41[4]; // [rsp+A0h] [rbp-40h] BYREF

  v36[0] = 0;
  v12 = *(_QWORD *)(a1 + 56);
  v13 = *(_DWORD *)(a1 + 64) == 372;
  v35[0] = (unsigned __int64)v36;
  v35[1] = 0;
  if ( v13 )
  {
    sub_2240AE0(v35, (unsigned __int64 *)(a1 + 72));
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v14 = sub_389B1F0((__int64)a2, (__int64)v35, v12);
  if ( !v14 )
  {
    sub_8FD6D0((__int64)v37, "unable to create block named '", v35);
    if ( v37[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v28 = (__m128i *)sub_2241490((unsigned __int64 *)v37, "'", 1u);
    v39 = v41;
    if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
    {
      v41[0] = _mm_loadu_si128(v28 + 1);
    }
    else
    {
      v39 = (_OWORD *)v28->m128i_i64[0];
      *(_QWORD *)&v41[0] = v28[1].m128i_i64[0];
    }
    v40 = v28->m128i_u64[1];
    v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
    v28->m128i_i64[1] = 0;
    v28[1].m128i_i8[0] = 0;
    v34 = 260;
    v33 = &v39;
    v23 = sub_38814C0(a1 + 8, v12, (__int64)&v33);
    if ( v39 != v41 )
      j_j___libc_free_0((unsigned __int64)v39);
    if ( (__int64 *)v37[0] != &v38 )
      j_j___libc_free_0(v37[0]);
    goto LABEL_19;
  }
  LOBYTE(v41[0]) = 0;
  v15 = 0;
  v39 = v41;
  v40 = 0;
  v31 = a1 + 8;
  while ( 1 )
  {
    v16 = *(_QWORD *)(a1 + 56);
    sub_2241130((unsigned __int64 *)&v39, 0, v15, byte_3F871B3, 0);
    v17 = *(_DWORD *)(a1 + 64);
    if ( v17 != 369 )
      break;
    v32 = *(_DWORD *)(a1 + 104);
    *(_DWORD *)(a1 + 64) = sub_3887100(v31);
    if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after instruction id") )
      goto LABEL_16;
LABEL_8:
    v18 = sub_38B1900(a1, v37, (__int64)v14, a2, *(double *)a3.m128_u64, a4, a5);
    if ( v18 == 1 )
      goto LABEL_16;
    v30 = v14 + 5;
    v19 = (__int64)(v14 + 5);
    v29 = v37[0];
    if ( v18 == 2 )
    {
      sub_157E9D0(v19, v37[0]);
      v26 = v14[5] & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v29 + 32) = v30;
      *(_QWORD *)(v29 + 24) = v26 | *(_QWORD *)(v29 + 24) & 7LL;
      *(_QWORD *)(v26 + 8) = v29 + 24;
      v14[5] = v14[5] & 7 | (v29 + 24);
    }
    else
    {
      sub_157E9D0(v19, v37[0]);
      v22 = v14[5] & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v29 + 32) = v30;
      *(_QWORD *)(v29 + 24) = v22 | *(_QWORD *)(v29 + 24) & 7LL;
      *(_QWORD *)(v22 + 8) = v29 + 24;
      v14[5] = v14[5] & 7 | (v29 + 24);
      if ( *(_DWORD *)(a1 + 64) != 4 )
        goto LABEL_11;
      *(_DWORD *)(a1 + 64) = sub_3887100(v31);
    }
    if ( (unsigned __int8)sub_38AA430(a1, v37[0], a3, *(double *)a4.m128i_i64, a5, a6, v24, v25, a9, a10) )
      goto LABEL_16;
LABEL_11:
    v23 = sub_38943C0(
            a2,
            v32,
            (unsigned __int64 *)&v39,
            v16,
            v37[0],
            a3,
            *(double *)a4.m128i_i64,
            a5,
            a6,
            v20,
            v21,
            a9,
            a10);
    if ( (_BYTE)v23 )
      goto LABEL_16;
    if ( (unsigned int)*(unsigned __int8 *)(v37[0] + 16) - 25 <= 9 )
      goto LABEL_17;
    v15 = v40;
  }
  if ( v17 != 375
    || (sub_2240AE0((unsigned __int64 *)&v39, (unsigned __int64 *)(a1 + 72)),
        *(_DWORD *)(a1 + 64) = sub_3887100(v31),
        !(unsigned __int8)sub_388AF10(a1, 3, "expected '=' after instruction name")) )
  {
    v32 = -1;
    goto LABEL_8;
  }
LABEL_16:
  v23 = 1;
LABEL_17:
  if ( v39 != v41 )
    j_j___libc_free_0((unsigned __int64)v39);
LABEL_19:
  if ( (_BYTE *)v35[0] != v36 )
    j_j___libc_free_0(v35[0]);
  return v23;
}
