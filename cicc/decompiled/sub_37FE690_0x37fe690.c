// Function: sub_37FE690
// Address: 0x37fe690
//
void __fastcall sub_37FE690(__int64 *a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5, __m128i a6)
{
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 (__fastcall **v16)(__int64, __int64, unsigned int); // rcx
  __int64 v17; // [rsp+0h] [rbp-D0h]
  __int64 v18; // [rsp+8h] [rbp-C8h]
  char v20; // [rsp+17h] [rbp-B9h]
  _WORD *v21; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+20h] [rbp-B0h] BYREF
  int v23; // [rsp+28h] [rbp-A8h]
  __m128i v24; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v25; // [rsp+40h] [rbp-90h]
  __int64 v26[4]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v27; // [rsp+70h] [rbp-60h]
  __int64 v28; // [rsp+78h] [rbp-58h]
  __int64 v29; // [rsp+80h] [rbp-50h]
  __int64 v30; // [rsp+88h] [rbp-48h]
  __int64 v31; // [rsp+90h] [rbp-40h]

  v9 = *(_DWORD *)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 40);
  if ( v9 > 239 )
  {
    if ( (unsigned int)(v9 - 242) > 1 )
    {
LABEL_4:
      a6 = _mm_loadu_si128((const __m128i *)v10);
      v11 = 0;
      v20 = 0;
      v24 = a6;
      v25 = _mm_loadu_si128((const __m128i *)(v10 + 40));
      v12 = 0;
      goto LABEL_5;
    }
  }
  else if ( v9 <= 237 && (unsigned int)(v9 - 101) > 0x2F )
  {
    goto LABEL_4;
  }
  v20 = 1;
  v24 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v25 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v11 = *(_QWORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
LABEL_5:
  v13 = *a1;
  v14 = *(_QWORD *)(a2 + 80);
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  LOBYTE(v31) = 4;
  v21 = (_WORD *)v13;
  v22 = v14;
  if ( v14 )
  {
    v17 = v11;
    v18 = v12;
    sub_B96E90((__int64)&v22, v14, 1);
    v11 = v17;
    v12 = v18;
  }
  v15 = a1[1];
  v16 = *(__int64 (__fastcall ***)(__int64, __int64, unsigned int))(a2 + 48);
  v23 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v26,
    v21,
    v15,
    a3,
    *(unsigned __int16 *)v16,
    v16[1],
    (__int64)&v24,
    2u,
    v27,
    v28,
    v29,
    v30,
    v31,
    (__int64)&v22,
    v11,
    v12);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  if ( v20 )
    sub_3760E70((__int64)a1, a2, 1, v26[2], v26[3]);
  sub_375AEA0(a1, v26[0], v26[1], a4, a5, a6);
}
