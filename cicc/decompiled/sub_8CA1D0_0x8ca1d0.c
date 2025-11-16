// Function: sub_8CA1D0
// Address: 0x8ca1d0
//
__int64 __fastcall sub_8CA1D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rsi

  sub_8C9930(a1, a2);
  result = *(unsigned __int8 *)(a2 + 80);
  if ( (unsigned __int8)(result - 3) > 2u )
  {
    if ( (unsigned __int8)(result - 10) <= 1u || (_BYTE)result == 17 )
    {
      v5 = *(_QWORD *)(a2 + 88);
      if ( !*(_QWORD *)(v5 + 32) )
        return (__int64)sub_8C7090(11, v5);
    }
    else
    {
      result = (unsigned int)(result - 7);
      if ( (result & 0xFD) == 0 )
      {
        v4 = *(_QWORD *)(a2 + 88);
        if ( v4 )
        {
          if ( (*(_BYTE *)(v4 + 170) & 0x10) != 0 )
          {
            result = *(_QWORD *)(v4 + 216);
            if ( *(_QWORD *)result )
            {
              if ( !*(_QWORD *)(v4 + 32) )
                return (__int64)sub_8C7090(7, v4);
            }
          }
        }
      }
    }
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 88);
    if ( !*(_QWORD *)(v3 + 32) )
      return sub_8CA0A0(v3, 1u);
  }
  return result;
}
