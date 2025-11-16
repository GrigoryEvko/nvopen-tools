// Function: sub_25BCA40
// Address: 0x25bca40
//
__int64 __fastcall sub_25BCA40(int *a1, unsigned __int8 **a2, char a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int8 v7; // bl
  unsigned __int8 *v8; // rdi
  int v9; // edx

  result = sub_CF5020(a4, (__int64)a2, 1u);
  v7 = result & a3;
  if ( v7 )
  {
    v8 = sub_98B9F0(*a2);
    result = *v8;
    if ( (unsigned __int8)result <= 0x1Cu )
    {
      if ( (_BYTE)result == 22 )
      {
        *a1 |= v7;
        return result;
      }
    }
    else if ( (_BYTE)result == 60 )
    {
      return result;
    }
    if ( (unsigned __int8)sub_CF7060(v8) )
      v9 = *a1;
    else
      v9 = v7 | *a1;
    result = v7 << 6;
    *a1 = v9 | result | (16 * v7);
  }
  return result;
}
