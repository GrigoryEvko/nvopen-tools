// Function: sub_3829070
// Address: 0x3829070
//
unsigned __int64 __fastcall sub_3829070(_QWORD *a1, unsigned __int64 a2)
{
  int v3; // eax
  bool v4; // dl
  __int64 v5; // r14
  const __m128i *v6; // rax
  __int64 v7; // r15
  __m128i v8; // xmm0
  unsigned __int16 *v9; // rax
  unsigned int v10; // ebx
  int v11; // eax
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int); // r9
  int v13; // ecx
  __int64 v14; // rsi
  _WORD *v15; // r11
  __int64 v16; // rsi
  __int64 *v18; // rax
  _WORD *v19; // [rsp+8h] [rbp-B8h]
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-B0h]
  int v21; // [rsp+18h] [rbp-A8h]
  char v22; // [rsp+1Fh] [rbp-A1h]
  __m128i v23; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v24; // [rsp+30h] [rbp-90h] BYREF
  int v25; // [rsp+38h] [rbp-88h]
  unsigned __int64 v26[4]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v27; // [rsp+60h] [rbp-60h]
  __int64 v28; // [rsp+68h] [rbp-58h]
  __int64 v29; // [rsp+70h] [rbp-50h]
  __int64 v30; // [rsp+78h] [rbp-48h]
  __int64 v31; // [rsp+80h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 > 239 )
  {
    if ( (unsigned int)(v3 - 242) > 1 )
      goto LABEL_16;
  }
  else
  {
    if ( v3 <= 237 && (unsigned int)(v3 - 101) > 0x2F )
    {
      if ( v3 == 220 )
      {
        v4 = 1;
LABEL_6:
        v22 = 0;
        v5 = 0;
        v6 = *(const __m128i **)(a2 + 40);
        v7 = 0;
        goto LABEL_7;
      }
LABEL_16:
      v4 = v3 == 143;
      goto LABEL_6;
    }
    if ( v3 == 220 )
    {
      v4 = 1;
      goto LABEL_20;
    }
  }
  v4 = v3 == 143;
LABEL_20:
  v18 = *(__int64 **)(a2 + 40);
  v22 = 1;
  v5 = *v18;
  v7 = v18[1];
  v6 = (const __m128i *)(v18 + 5);
LABEL_7:
  v8 = _mm_loadu_si128(v6);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v23 = v8;
  v10 = *v9;
  v20 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v9 + 1);
  if ( v4 )
    v11 = sub_2FE5C30(
            *(_WORD *)(*(_QWORD *)(v8.m128i_i64[0] + 48) + 16LL * v8.m128i_u32[2]),
            *(_QWORD *)(*(_QWORD *)(v8.m128i_i64[0] + 48) + 16LL * v8.m128i_u32[2] + 8),
            v10);
  else
    v11 = sub_2FE5D50(
            *(_WORD *)(*(_QWORD *)(v8.m128i_i64[0] + 48) + 16LL * v8.m128i_u32[2]),
            *(_QWORD *)(*(_QWORD *)(v8.m128i_i64[0] + 48) + 16LL * v8.m128i_u32[2] + 8),
            v10);
  v12 = v20;
  v13 = v11;
  v14 = *(_QWORD *)(a2 + 80);
  v27 = 0;
  v28 = 0;
  v15 = (_WORD *)*a1;
  v29 = 0;
  v30 = 0;
  LOBYTE(v31) = 5;
  v24 = v14;
  if ( v14 )
  {
    v21 = v11;
    v19 = v15;
    sub_B96E90((__int64)&v24, v14, 1);
    v12 = v20;
    v13 = v21;
    v15 = v19;
  }
  v16 = a1[1];
  v25 = *(_DWORD *)(a2 + 72);
  sub_3494590((__int64)v26, v15, v16, v13, v10, v12, (__int64)&v23, 1u, v27, v28, v29, v30, v31, (__int64)&v24, v5, v7);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  if ( !v22 )
    return v26[0];
  sub_3760E70((__int64)a1, a2, 1, v26[2], v26[3]);
  sub_3760E70((__int64)a1, a2, 0, v26[0], v26[1]);
  return 0;
}
