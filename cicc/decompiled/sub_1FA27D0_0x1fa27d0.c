// Function: sub_1FA27D0
// Address: 0x1fa27d0
//
__int64 __fastcall sub_1FA27D0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  const __m128i *v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int8 *v11; // rax
  const void **v12; // rcx
  unsigned int v13; // r15d
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // r9
  int v17; // edx
  int v18; // eax
  __int64 result; // rax
  int v20; // r8d
  _QWORD *v21; // rax
  int v22; // edx
  unsigned __int32 v23; // edx
  __int128 v24; // [rsp-20h] [rbp-B0h]
  __int128 v25; // [rsp-10h] [rbp-A0h]
  unsigned __int32 v26; // [rsp+0h] [rbp-90h]
  int v27; // [rsp+8h] [rbp-88h]
  const void **v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+18h] [rbp-78h]
  _QWORD *v30; // [rsp+18h] [rbp-78h]
  __int64 v31; // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  int v33; // [rsp+38h] [rbp-58h]
  __int64 *v34; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int32 v35; // [rsp+48h] [rbp-48h]
  _QWORD *v36; // [rsp+50h] [rbp-40h]
  int v37; // [rsp+58h] [rbp-38h]

  v6 = *(const __m128i **)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = _mm_loadu_si128(v6);
  v9 = v6[2].m128i_i64[1];
  v10 = v6[3].m128i_i64[0];
  v29 = v6->m128i_i64[0];
  v26 = v6->m128i_u32[2];
  v11 = (unsigned __int8 *)(*(_QWORD *)(v6->m128i_i64[0] + 40) + 16LL * v26);
  v12 = (const void **)*((_QWORD *)v11 + 1);
  v13 = *v11;
  v32 = v7;
  v28 = v12;
  if ( v7 )
    sub_1623A60((__int64)&v32, v7, 2);
  v33 = *(_DWORD *)(a2 + 64);
  if ( !(unsigned __int8)sub_1D18C40(a2, 1) )
  {
LABEL_15:
    v21 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v32, 0x6Fu, 0, v14);
    *((_QWORD *)&v25 + 1) = v10;
    *(_QWORD *)&v25 = v9;
    v27 = v22;
    v30 = v21;
    v34 = sub_1D332F0(
            *a1,
            52,
            (__int64)&v32,
            v13,
            v28,
            0,
            *(double *)v8.m128i_i64,
            a4,
            a5,
            v8.m128i_i64[0],
            v8.m128i_u64[1],
            v25);
    v35 = v23;
    v36 = v30;
    v37 = v27;
    goto LABEL_16;
  }
  v15 = *(unsigned __int16 *)(v29 + 24);
  if ( v15 == 32 || v15 == 10 )
  {
    v18 = *(unsigned __int16 *)(v9 + 24);
    if ( v18 != 32 && v18 != 10 )
    {
      *((_QWORD *)&v24 + 1) = v10;
      *(_QWORD *)&v24 = v9;
      result = (__int64)sub_1D37440(
                          *a1,
                          64,
                          (__int64)&v32,
                          *(const void ****)(a2 + 40),
                          *(_DWORD *)(a2 + 60),
                          v14,
                          *(double *)v8.m128i_i64,
                          a4,
                          a5,
                          v24,
                          *(_OWORD *)&v8);
      goto LABEL_11;
    }
  }
  if ( !sub_1D185B0(v9) )
  {
    v20 = sub_1D1FED0((__int64)*a1, v8.m128i_i64[0], v8.m128i_i64[1], v9, v10);
    result = 0;
    if ( v20 )
      goto LABEL_11;
    goto LABEL_15;
  }
  v36 = sub_1D2B300(*a1, 0x3Fu, (__int64)&v32, 0x6Fu, 0, v16);
  v34 = (__int64 *)v29;
  v37 = v17;
  v35 = v26;
LABEL_16:
  result = sub_1F994A0((__int64)a1, a2, (__int64 *)&v34, 2, 1);
LABEL_11:
  if ( v32 )
  {
    v31 = result;
    sub_161E7C0((__int64)&v32, v32);
    return v31;
  }
  return result;
}
