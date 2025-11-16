// Function: sub_21E5090
// Address: 0x21e5090
//
__int64 __fastcall sub_21E5090(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  unsigned __int8 v7; // bl
  __int64 v8; // rax
  _QWORD *v9; // rsi
  __int32 v10; // edx
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rcx
  const void **v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  const __m128i *v17; // r8
  __m128i v18; // xmm2
  __int32 v19; // edx
  __m128i v20; // xmm1
  __m128i si128; // xmm4
  __m128i v22; // xmm0
  __m128i v23; // xmm3
  __int64 v24; // r9
  unsigned int v25; // ecx
  __int64 v26; // rax
  unsigned int v27; // edx
  __int64 v28; // r14
  const __m128i *v29; // r8
  __int64 v30; // r9
  int v31; // r8d
  unsigned int v32; // eax
  __int16 v33; // si
  __int64 v34; // r14
  __int64 v36; // [rsp+0h] [rbp-2F0h]
  __int32 v37; // [rsp+8h] [rbp-2E8h]
  const __m128i *v38; // [rsp+8h] [rbp-2E8h]
  __int64 v39; // [rsp+10h] [rbp-2E0h]
  const __m128i *v40; // [rsp+10h] [rbp-2E0h]
  _QWORD *v41; // [rsp+18h] [rbp-2D8h]
  __int64 v42; // [rsp+20h] [rbp-2D0h] BYREF
  int v43; // [rsp+28h] [rbp-2C8h]
  __int64 v44; // [rsp+30h] [rbp-2C0h] BYREF
  int v45; // [rsp+38h] [rbp-2B8h]
  __m128i v46; // [rsp+40h] [rbp-2B0h] BYREF
  __m128i v47; // [rsp+50h] [rbp-2A0h] BYREF
  __m128i v48; // [rsp+60h] [rbp-290h] BYREF
  __m128i v49[4]; // [rsp+70h] [rbp-280h] BYREF
  __int64 *v50; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v51; // [rsp+B8h] [rbp-238h]
  _OWORD v52[35]; // [rsp+C0h] [rbp-230h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v42 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v42, v6, 2);
  v7 = *(_BYTE *)(a2 + 88);
  v43 = *(_DWORD *)(a2 + 64);
  v41 = *(_QWORD **)(a1 - 176);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 88LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v39 = sub_1D38BB0((__int64)v41, (__int64)v9, (__int64)&v42, 6, 0, 1, a3, a4, a5, 0);
  v37 = v10;
  sub_21E3B00(
    &v46,
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL),
    a3,
    a4,
    a5);
  v11 = *(_QWORD *)(a2 + 72);
  v44 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v44, v11, 2);
  v45 = *(_DWORD *)(a2 + 64);
  v12 = sub_21DEF90(a2);
  v16 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v12, (__int64)&v44, v13, v14, v15, a3, a4, a5, 0);
  v17 = *(const __m128i **)(a2 + 32);
  v50 = (__int64 *)v52;
  v49[0].m128i_i64[0] = v16;
  v18 = _mm_load_si128(&v46);
  v49[0].m128i_i32[2] = v19;
  v48.m128i_i64[0] = v39;
  v20 = _mm_load_si128(&v47);
  si128 = _mm_load_si128(v49);
  v49[1] = v18;
  v48.m128i_i32[2] = v37;
  v22 = _mm_loadu_si128(v17 + 10);
  v23 = _mm_load_si128(&v48);
  v51 = 0x2000000005LL;
  v49[2] = v20;
  v49[3] = v22;
  v52[0] = v23;
  v52[1] = si128;
  v52[2] = v18;
  v52[3] = v20;
  v52[4] = v22;
  if ( !v44 )
  {
    v25 = *(_DWORD *)(a2 + 56);
    if ( v25 <= 6 )
    {
      v26 = 5;
      goto LABEL_15;
    }
    v27 = 32;
    v26 = 5;
    goto LABEL_9;
  }
  sub_161E7C0((__int64)&v44, v44);
  v25 = *(_DWORD *)(a2 + 56);
  v17 = *(const __m128i **)(a2 + 32);
  v26 = (unsigned int)v51;
  v27 = HIDWORD(v51);
  if ( v25 > 6 )
  {
LABEL_9:
    v28 = 240;
    v24 = 40LL * (v25 - 7) + 280;
    do
    {
      v29 = (const __m128i *)((char *)v17 + v28);
      if ( (unsigned int)v26 >= v27 )
      {
        v36 = v24;
        v38 = v29;
        sub_16CD150((__int64)&v50, v52, 0, 16, (int)v29, v24);
        v26 = (unsigned int)v51;
        v24 = v36;
        v29 = v38;
      }
      v28 += 40;
      *(__m128i *)&v50[2 * v26] = _mm_loadu_si128(v29);
      v17 = *(const __m128i **)(a2 + 32);
      v27 = HIDWORD(v51);
      v26 = (unsigned int)(v51 + 1);
      LODWORD(v51) = v51 + 1;
    }
    while ( v24 != v28 );
  }
  if ( v27 <= (unsigned int)v26 )
  {
    v40 = v17;
    sub_16CD150((__int64)&v50, v52, 0, 16, (int)v17, v24);
    v26 = (unsigned int)v51;
    v17 = v40;
  }
LABEL_15:
  *(__m128i *)&v50[2 * v26] = _mm_loadu_si128(v17);
  v30 = *(_QWORD *)(a2 + 40);
  v31 = *(_DWORD *)(a2 + 60);
  v32 = v51 + 1;
  LODWORD(v51) = v51 + 1;
  if ( v7 > 0x2Cu )
  {
    v33 = 3165;
    if ( v7 != 92 )
      v33 = 3 * (v7 == 95) + 3164;
  }
  else
  {
    switch ( v7 )
    {
      case 5u:
        v33 = 3170;
        break;
      case 6u:
      case 7u:
      case 8u:
      case 9u:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0xFu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x21u:
      case 0x22u:
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x29u:
      case 0x2Cu:
        v33 = 3173;
        break;
      case 0xAu:
        v33 = 3166;
        break;
      case 0x2Au:
        v33 = 3171;
        break;
      case 0x2Bu:
        v33 = 3172;
        break;
    }
  }
  v34 = sub_1D23DE0(v41, v33, (__int64)&v42, v30, v31, v30, v50, v32);
  if ( v50 != (__int64 *)v52 )
    _libc_free((unsigned __int64)v50);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  return v34;
}
