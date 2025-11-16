// Function: sub_135CB20
// Address: 0x135cb20
//
__int64 __fastcall sub_135CB20(__int64 a1, __int64 a2)
{
  __m128i v3; // [rsp+0h] [rbp-40h] BYREF
  __int64 v4; // [rsp+10h] [rbp-30h]

  v3 = 0u;
  v4 = 0;
  sub_14A8180(a2, &v3, 0);
  return sub_135C460(a1, *(_QWORD *)(a2 - 24), 0xFFFFFFFFFFFFFFFFLL, &v3, 3);
}
