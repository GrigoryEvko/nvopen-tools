// Function: sub_21E47B0
// Address: 0x21e47b0
//
__int64 __fastcall sub_21E47B0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  unsigned __int8 v7; // al
  __int16 v8; // r14
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rax
  int v12; // edx
  int v13; // r15d
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rcx
  const void **v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // r8d
  const __m128i *v22; // rax
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  int v25; // edx
  __m128i v26; // xmm2
  __int64 v27; // r9
  __int64 v28; // r12
  __int64 v30; // [rsp+0h] [rbp-E0h]
  _QWORD *v31; // [rsp+8h] [rbp-D8h]
  __int64 v32; // [rsp+10h] [rbp-D0h] BYREF
  int v33; // [rsp+18h] [rbp-C8h]
  __int64 v34; // [rsp+20h] [rbp-C0h] BYREF
  int v35; // [rsp+28h] [rbp-B8h]
  __m128i v36; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+50h] [rbp-90h] BYREF
  int v39; // [rsp+58h] [rbp-88h]
  __int64 v40; // [rsp+60h] [rbp-80h]
  int v41; // [rsp+68h] [rbp-78h]
  __m128i v42; // [rsp+70h] [rbp-70h]
  __m128i v43; // [rsp+80h] [rbp-60h]
  __m128i v44; // [rsp+90h] [rbp-50h]
  __m128i v45; // [rsp+A0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  v32 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v32, v6, 2);
  v33 = *(_DWORD *)(a2 + 64);
  v31 = *(_QWORD **)(a1 - 176);
  v7 = *(_BYTE *)(a2 + 88);
  if ( v7 > 0x2Cu )
  {
    v8 = 3153;
    if ( v7 != 92 )
      v8 = 3 * (v7 == 95) + 3152;
  }
  else
  {
    switch ( v7 )
    {
      case 5u:
        v8 = 3158;
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
        v8 = 3161;
        break;
      case 0xAu:
        v8 = 3154;
        break;
      case 0x2Au:
        v8 = 3159;
        break;
      case 0x2Bu:
        v8 = 3160;
        break;
    }
  }
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 88LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = sub_1D38BB0((__int64)v31, (__int64)v10, (__int64)&v32, 6, 0, 1, a3, a4, a5, 0);
  v13 = v12;
  v30 = v11;
  sub_21E3B00(
    &v36,
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL),
    a3,
    a4,
    a5);
  v14 = *(_QWORD *)(a2 + 72);
  v39 = v13;
  v38 = v30;
  v34 = v14;
  if ( v14 )
    sub_1623A60((__int64)&v34, v14, 2);
  v35 = *(_DWORD *)(a2 + 64);
  v15 = sub_21DEF90(a2);
  v19 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v15, (__int64)&v34, v16, v17, v18, a3, a4, a5, 0);
  v20 = *(_QWORD *)(a2 + 40);
  v21 = *(_DWORD *)(a2 + 60);
  v40 = v19;
  v22 = *(const __m128i **)(a2 + 32);
  v23 = _mm_loadu_si128(&v36);
  v24 = _mm_loadu_si128(&v37);
  v41 = v25;
  v26 = _mm_loadu_si128(v22 + 10);
  v42 = v23;
  v43 = v24;
  v44 = v26;
  v45 = _mm_loadu_si128(v22);
  v28 = sub_1D23DE0(v31, v8, (__int64)&v32, v20, v21, v27, &v38, 6);
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return v28;
}
