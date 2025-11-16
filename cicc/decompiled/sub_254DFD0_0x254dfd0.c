// Function: sub_254DFD0
// Address: 0x254dfd0
//
__int64 __fastcall sub_254DFD0(__int64 a1, __int64 a2)
{
  __m128i *v2; // r15
  char v3; // al
  __int64 v4; // r8
  unsigned __int64 v5; // r8
  char v7; // [rsp+Bh] [rbp-55h] BYREF
  unsigned int v8; // [rsp+Ch] [rbp-54h] BYREF
  _QWORD v9[10]; // [rsp+10h] [rbp-50h] BYREF

  v2 = (__m128i *)(a1 + 72);
  LODWORD(v9[0]) = 19;
  if ( (unsigned __int8)sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)v9, 1, 0, 0)
    && (unsigned __int8)sub_252A800(a2, v2, a1, (bool *)v9) )
  {
    return 1;
  }
  v3 = sub_2509800(v2);
  v4 = *(_QWORD *)(a1 + 72);
  v7 = v3;
  if ( (v4 & 3) == 3 )
    v5 = *(_QWORD *)((v4 & 0xFFFFFFFFFFFFFFFCLL) + 24);
  else
    v5 = v4 & 0xFFFFFFFFFFFFFFFCLL;
  v9[0] = &v7;
  v8 = 1;
  v9[1] = v5;
  v9[2] = a2;
  v9[3] = a1;
  v9[4] = &v8;
  v9[5] = a1 + 88;
  if ( (unsigned __int8)sub_251CC40(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64 *, __int64))sub_259AE80,
                          (__int64)v9,
                          a1,
                          v5) )
    return v8;
  else
    return sub_2505E20(a1 + 88);
}
