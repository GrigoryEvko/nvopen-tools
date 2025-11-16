// Function: sub_D5CBF0
// Address: 0xd5cbf0
//
__int64 __fastcall sub_D5CBF0(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __m128i v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 v6; // [rsp+18h] [rbp-18h]

  v3 = sub_D5BAA0(a1);
  result = 0;
  if ( v3 )
  {
    if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL) + 8LL) == 14 )
    {
      sub_D5BC90(&v5, v3, 3u, a2);
      return v6;
    }
  }
  return result;
}
