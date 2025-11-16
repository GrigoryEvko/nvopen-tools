// Function: sub_12FB780
// Address: 0x12fb780
//
unsigned __int64 *__fastcall sub_12FB780(
        unsigned __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5)
{
  unsigned __int64 *result; // rax
  __int64 v6; // rdi
  int v7; // r9d
  char v8; // r9
  unsigned __int64 v9; // r10

  result = a1;
  v6 = a4;
  v7 = -(int)a5;
  if ( a5 > 0x3F )
  {
    v9 = 0;
    if ( a5 != 64 )
    {
      v6 = a3 | a4;
      if ( a5 <= 0x7F )
      {
        a3 = a2 << v7;
        a2 >>= a5;
      }
      else
      {
        if ( a5 == 128 )
          a3 = a2;
        else
          a3 = a2 != 0;
        a2 = 0;
      }
    }
  }
  else
  {
    v8 = v7 & 0x3F;
    v9 = a2 >> a5;
    a2 = (a3 >> a5) | (a2 << v8);
    a3 <<= v8;
  }
  result[1] = a2;
  result[2] = v9;
  *result = (v6 != 0) | a3;
  return result;
}
