// Function: sub_D5CAE0
// Address: 0xd5cae0
//
bool __fastcall sub_D5CAE0(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v2; // rax
  bool result; // al
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  bool v5; // [rsp+18h] [rbp-18h]

  v2 = sub_D5BAA0(a1);
  if ( !v2 )
    return (sub_D5BB80(a1) & 3) != 0;
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v2 + 24) + 16LL) + 8LL) != 14 )
    return (sub_D5BB80(a1) & 3) != 0;
  sub_D5BC90(&v4, v2, 7u, a2);
  result = v5;
  if ( !v5 )
    return (sub_D5BB80(a1) & 3) != 0;
  return result;
}
