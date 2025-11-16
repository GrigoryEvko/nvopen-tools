// Function: sub_1562E30
// Address: 0x1562e30
//
__m128i *__fastcall sub_1562E30(__m128i *a1, __int64 a2)
{
  int v2; // ecx
  _BYTE *v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rbx
  _BYTE *v7; // rax
  __int64 v8; // rdx
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v9[0] = a2;
  if ( sub_155D3E0((__int64)v9) )
  {
    v4 = (_BYTE *)sub_155D8B0(v9);
    v6 = v5;
    v7 = (_BYTE *)sub_155D7D0(v9);
    sub_1562A10(a1, v7, v8, v4, v6);
    return a1;
  }
  else
  {
    v2 = sub_155D410(v9);
    a1->m128i_i64[0] |= 1LL << v2;
    if ( v2 == 1 )
    {
      a1[3].m128i_i64[1] = (unsigned int)sub_155D730(v9);
      return a1;
    }
    else if ( v2 == 48 )
    {
      a1[4].m128i_i64[0] = (unsigned int)sub_155D7A0(v9);
      return a1;
    }
    else
    {
      switch ( v2 )
      {
        case 9:
          a1[4].m128i_i64[1] = sub_155D740(v9);
          break;
        case 10:
          a1[5].m128i_i64[0] = sub_155D7B0(v9);
          break;
        case 2:
          a1[5].m128i_i64[1] = sub_155D4B0(v9);
          break;
      }
      return a1;
    }
  }
}
