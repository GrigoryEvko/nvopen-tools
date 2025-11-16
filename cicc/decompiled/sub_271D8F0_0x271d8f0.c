// Function: sub_271D8F0
// Address: 0x271d8f0
//
__int64 __fastcall sub_271D8F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // bl
  __int64 result; // rax
  unsigned __int8 v7; // [rsp+Fh] [rbp-11h]

  v5 = *(_BYTE *)(a1 + 2);
  result = sub_3181330(a2, a3, a4, a5);
  if ( (_BYTE)result )
  {
    if ( v5 == 3 )
    {
      v7 = result;
      sub_271D2E0(a1, 2);
      return v7;
    }
    if ( v5 > 3u )
    {
      if ( (unsigned __int8)(v5 - 4) <= 1u )
        return 0;
    }
    else if ( (v5 & 1) == 0 )
    {
      return 0;
    }
    BUG();
  }
  return result;
}
