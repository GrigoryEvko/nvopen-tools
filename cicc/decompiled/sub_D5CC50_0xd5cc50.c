// Function: sub_D5CC50
// Address: 0xd5cc50
//
__int64 __fastcall sub_D5CC50(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 v5; // [rsp+18h] [rbp-18h]

  v2 = sub_D5BAA0(a1);
  if ( !v2 )
    return sub_D5BB80(a1) & 1;
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v2 + 24) + 16LL) + 8LL) != 14 )
    return sub_D5BB80(a1) & 1;
  sub_D5BC90(&v4, v2, 7u, a2);
  result = v5;
  if ( !v5 )
    return sub_D5BB80(a1) & 1;
  return result;
}
