// Function: sub_2597270
// Address: 0x2597270
//
bool __fastcall sub_2597270(__int64 *a1, unsigned __int8 *a2)
{
  int v3; // eax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __m128i v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdi
  char v12; // [rsp+6h] [rbp-2Ah] BYREF
  bool v13; // [rsp+7h] [rbp-29h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h] BYREF
  __m128i v15[2]; // [rsp+10h] [rbp-20h] BYREF

  v3 = *a2;
  if ( (unsigned __int8)v3 <= 0x15u )
  {
    if ( sub_AC30F0((__int64)a2) )
      return 1;
    v3 = *a2;
    if ( (unsigned int)(v3 - 12) <= 1 )
      return 1;
  }
  if ( (unsigned __int8)v3 > 0x1Cu )
  {
    v4 = (unsigned int)(v3 - 34);
    if ( (unsigned __int8)v4 <= 0x33u )
    {
      v5 = 0x8000000000041LL;
      if ( _bittest64(&v5, v4) )
      {
        v6.m128i_i64[0] = sub_250D2C0((unsigned __int64)a2, 0);
        v7 = a1[1];
        v8 = *a1;
        v15[0] = v6;
        if ( (unsigned __int8)sub_2596DB0(v8, v7, v15, 0, &v12, 0, 0) )
        {
          v9 = a1[1];
          v10 = *a1;
          v14 = 0;
          if ( !(unsigned __int8)sub_25890A0(v10, v9, v15[0].m128i_i64, 0, &v13, 0, &v14) )
          {
            if ( v14 )
              return (*(_WORD *)(v14 + 98) & 3) == 3;
            return 0;
          }
          return 1;
        }
      }
    }
  }
  return 0;
}
