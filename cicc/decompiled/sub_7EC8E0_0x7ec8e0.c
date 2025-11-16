// Function: sub_7EC8E0
// Address: 0x7ec8e0
//
__int64 __fastcall sub_7EC8E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 result; // rax
  __m128i v4[7]; // [rsp+0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a1 + 184);
  sub_7F9080(a1, v4);
  result = sub_7FEC50(v2, (unsigned int)v4, 0, 0, 1, 0, a2, 0, 0);
  if ( (*(_BYTE *)(a1 - 8) & 8) == 0 )
    return sub_7EC5C0(a1, v4);
  return result;
}
