// Function: sub_AF48C0
// Address: 0xaf48c0
//
__int64 __fastcall sub_AF48C0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // rsi
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  *a1 = sub_B12000(a2 + 72);
  v2 = sub_B11F60(a2 + 80);
  sub_AF47B0((__int64)(a1 + 1), *(unsigned __int64 **)(v2 + 16), *(unsigned __int64 **)(v2 + 24));
  v3 = *(_QWORD *)(a2 + 24);
  v6[0] = v3;
  if ( v3 )
    sub_B96E90(v6, v3, 1);
  result = sub_B10D40(v6);
  v5 = v6[0];
  a1[4] = result;
  if ( v5 )
    return sub_B91220(v6);
  return result;
}
