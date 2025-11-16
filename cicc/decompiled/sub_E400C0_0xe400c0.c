// Function: sub_E400C0
// Address: 0xe400c0
//
__int64 __fastcall sub_E400C0(unsigned __int8 *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 *v3; // rsi
  __int64 v4; // rcx
  int v5; // eax
  unsigned __int64 v6; // rax

  result = 0;
  if ( a2 )
  {
    v3 = &a1[a2];
    if ( a1 == v3 )
    {
      return 1;
    }
    else
    {
      v4 = 0x10FFFFFFE07FE001LL;
      while ( 1 )
      {
        v5 = *a1;
        if ( (unsigned __int8)(v5 - 97) > 0x19u )
        {
          v6 = (unsigned int)(v5 - 35);
          if ( (unsigned __int8)v6 > 0x3Cu || !_bittest64(&v4, v6) )
            break;
        }
        if ( v3 == ++a1 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
