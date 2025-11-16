// Function: sub_3805750
// Address: 0x3805750
//
__int64 *__fastcall sub_3805750(__int64 a1, unsigned __int64 a2)
{
  bool v2; // zf
  __int64 *v3; // r10
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r10
  unsigned __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // r14
  unsigned __int64 v12; // [rsp+0h] [rbp-80h]
  __int64 v13; // [rsp+8h] [rbp-78h]
  __int64 *v14; // [rsp+10h] [rbp-70h]
  const __m128i *v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  int v17; // [rsp+28h] [rbp-58h]
  __m128i v18; // [rsp+30h] [rbp-50h] BYREF
  __m128i v19; // [rsp+40h] [rbp-40h]

  v2 = *(_DWORD *)(a2 + 24) == 339;
  v3 = *(__int64 **)(a1 + 8);
  v15 = *(const __m128i **)(a2 + 112);
  v4 = *(_QWORD *)(a2 + 40);
  v18 = _mm_loadu_si128((const __m128i *)v4);
  if ( v2 )
    v19 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  else
    v19 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v14 = v3;
  v5 = sub_33E5110(v3, 6, 0, 1, 0);
  v7 = v14;
  v8 = v5;
  v9 = v6;
  v16 = *(_QWORD *)(a2 + 80);
  if ( v16 )
  {
    v13 = v6;
    v12 = v5;
    sub_B96E90((__int64)&v16, v16, 1);
    v8 = v12;
    v9 = v13;
    v7 = v14;
  }
  v17 = *(_DWORD *)(a2 + 72);
  v10 = sub_33E6BC0(v7, 338, (__int64)&v16, 6u, 0, v15, v8, v9, (unsigned __int64 *)&v18, 2);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v10, 1);
  return v10;
}
