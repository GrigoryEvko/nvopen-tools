// Function: sub_21E2840
// Address: 0x21e2840
//
__int64 __fastcall sub_21E2840(
        __int64 a1,
        __int64 a2,
        int a3,
        double a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rsi
  int v11; // r8d
  _QWORD *v12; // r12
  __int64 v13; // rdx
  _QWORD *v14; // rax
  unsigned int v15; // ecx
  char v16; // r13
  _BYTE *v17; // rax
  __int16 v19; // r15
  __int64 v20; // rax
  __m128i v21; // xmm0
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int32 v24; // edx
  __int64 v25; // rdx
  __int64 v26; // r12
  int v28; // [rsp+8h] [rbp-88h]
  unsigned int v30; // [rsp+Ch] [rbp-84h]
  char v31; // [rsp+Ch] [rbp-84h]
  __int64 v32; // [rsp+10h] [rbp-80h] BYREF
  int v33; // [rsp+18h] [rbp-78h]
  __int64 v34; // [rsp+20h] [rbp-70h] BYREF
  int v35; // [rsp+28h] [rbp-68h]
  __m128i v36; // [rsp+30h] [rbp-60h] BYREF
  __m128i v37; // [rsp+40h] [rbp-50h]
  __m128i v38; // [rsp+50h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x48u )
    goto LABEL_11;
  v10 = *(_QWORD *)(a2 + 72);
  v11 = a3;
  v12 = *(_QWORD **)(a1 - 176);
  v32 = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)&v32, v10, 2);
    v11 = a3;
  }
  v33 = *(_DWORD *)(a2 + 64);
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) + 88LL);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = (unsigned int)v14;
  v16 = (unsigned __int8)v14 & 0x1E;
  if ( ((unsigned __int8)v14 & 0x1E) != 0 )
  {
    v28 = v11;
    v30 = (unsigned int)v14;
    v17 = sub_16D40F0((__int64)qword_4FBB4B0);
    v15 = v30;
    v11 = v28;
    if ( !(v17 ? *v17 : LOBYTE(qword_4FBB4B0[2])) )
LABEL_11:
      sub_16BD130("Not supported on this architecture", 1u);
  }
  switch ( (v15 >> 1) & 0xF )
  {
    case 0u:
      v19 = 2926;
      if ( v11 != 1 )
        v19 = (v11 != 2) + 2927;
      goto LABEL_14;
    case 1u:
      v19 = 2917;
      if ( v11 != 1 )
        v19 = (v11 != 2) + 2918;
      goto LABEL_14;
    case 2u:
      v19 = 2914;
      if ( v11 != 1 )
        v19 = (v11 != 2) + 2915;
      goto LABEL_14;
    case 3u:
      v19 = 2923;
      if ( v11 != 1 )
        v19 = (v11 != 2) + 2924;
      goto LABEL_14;
    case 4u:
      v19 = 2920;
      if ( v11 == 1 )
      {
LABEL_14:
        v20 = *(_QWORD *)(a2 + 32);
        if ( v16 )
        {
LABEL_25:
          v25 = 2;
          v36 = _mm_loadu_si128((const __m128i *)(v20 + 120));
          v37 = _mm_loadu_si128((const __m128i *)v20);
          goto LABEL_20;
        }
      }
      else
      {
        v20 = *(_QWORD *)(a2 + 32);
        v19 = (v11 != 2) + 2921;
        if ( v16 )
          goto LABEL_25;
      }
      v21 = _mm_loadu_si128((const __m128i *)(v20 + 120));
      v22 = *(_QWORD *)(a2 + 72);
      v34 = v22;
      v36 = v21;
      if ( v22 )
      {
        v31 = v15;
        sub_1623A60((__int64)&v34, v22, 2);
        LOBYTE(v15) = v31;
      }
      v23 = *(_QWORD *)(a1 - 176);
      v35 = *(_DWORD *)(a2 + 64);
      v37.m128i_i64[0] = sub_1D38BB0(v23, v15 & 1, (__int64)&v34, 5, 0, 1, v21, a5, a6, 0);
      v37.m128i_i32[2] = v24;
      if ( v34 )
        sub_161E7C0((__int64)&v34, v34);
      v25 = 3;
      v38 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
LABEL_20:
      v26 = sub_1D23DE0(v12, v19, (__int64)&v32, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), a9, v36.m128i_i64, v25);
      if ( v32 )
        sub_161E7C0((__int64)&v32, v32);
      return v26;
    default:
      sub_16BD130("Unknown size", 1u);
  }
}
