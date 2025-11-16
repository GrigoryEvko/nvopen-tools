// Function: sub_7EBB70
// Address: 0x7ebb70
//
_BYTE *__fastcall sub_7EBB70(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rsi
  __m128i *v7; // [rsp+8h] [rbp-18h] BYREF

  if ( !(unsigned int)sub_7EBAB0(a1, &v7) )
    return sub_730690(a1);
  if ( (unsigned int)sub_8D3410(*(_QWORD *)(a1 + 128)) )
  {
    v1 = sub_731250((__int64)v7);
  }
  else
  {
    v3 = sub_73E830((__int64)v7);
    v1 = v3;
    v6 = v7[7].m128i_i64[1];
    if ( *v3 != v6 && !(unsigned int)sub_8D97D0(*v3, v6, 0, v4, v5) )
      v1[1] = v7[7].m128i_i64[1];
  }
  if ( (*(_BYTE *)(a1 + 168) & 8) != 0 )
    return sub_73E130(v1, *(_QWORD *)(a1 + 128));
  return v1;
}
