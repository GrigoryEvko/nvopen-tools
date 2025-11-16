// Function: sub_252C090
// Address: 0x252c090
//
__int64 __fastcall sub_252C090(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v5; // rdx
  __int64 v7; // rax
  __m128i v8[3]; // [rsp+0h] [rbp-30h] BYREF

  if ( (unsigned __int8)(*a2 - 34) <= 0x33u && (v5 = 0x8000000000041LL, _bittest64(&v5, (unsigned int)*a2 - 34)) )
  {
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 39)
      || (unsigned __int8)sub_B49560((__int64)a2, 39)
      || !(unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 6)
      && !(unsigned __int8)sub_B49560((__int64)a2, 6)
      && !(unsigned __int8)sub_B46420((__int64)a2)
      && !(unsigned __int8)sub_B46490((__int64)a2)
      || (unsigned __int8)sub_2553B40(a2) )
    {
      return 1;
    }
    sub_250D230((unsigned __int64 *)v8, (unsigned __int64)a2, 5, 0);
    v3 = sub_2516660(a1, v8, 0);
    if ( !(_BYTE)v3 )
    {
      v7 = sub_252BBE0(a1, v8[0].m128i_i64[0], v8[0].m128i_i64[1], a3, 1, 0, 1);
      if ( v7 )
        return *(unsigned __int8 *)(v7 + 97);
    }
  }
  else
  {
    if ( !(unsigned __int8)sub_B46420((__int64)a2) && !(unsigned __int8)sub_B46490((__int64)a2) )
      return 1;
    v3 = 0;
    if ( !sub_B46560(a2) )
      return (unsigned int)sub_2553A90(a2) ^ 1;
  }
  return v3;
}
