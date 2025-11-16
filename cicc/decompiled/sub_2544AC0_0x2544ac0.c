// Function: sub_2544AC0
// Address: 0x2544ac0
//
__int64 __fastcall sub_2544AC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  char v3; // r8
  __int64 result; // rax
  unsigned __int8 v5; // r12
  __m128i v6; // [rsp+0h] [rbp-60h] BYREF
  void (__fastcall *v7)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-50h]
  _BYTE v8[16]; // [rsp+20h] [rbp-40h] BYREF
  void (__fastcall *v9)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-30h]

  v2 = sub_250C680((__int64 *)(a1 + 72));
  v3 = sub_2523DA0(a2, v2);
  result = 1;
  if ( v3 )
  {
    v9 = 0;
    v7 = 0;
    v5 = sub_25139F0(a2, v2, 0, 0, &v6, (__int64)v8);
    if ( v7 )
      v7(&v6, &v6, 3);
    if ( v9 )
      v9(v8, v8, 3);
    return v5 ^ 1u;
  }
  return result;
}
