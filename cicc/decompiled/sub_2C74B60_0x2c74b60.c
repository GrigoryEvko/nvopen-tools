// Function: sub_2C74B60
// Address: 0x2c74b60
//
__int64 __fastcall sub_2C74B60(unsigned __int64 *a1, unsigned int a2, unsigned int a3)
{
  unsigned int v4; // r12d

  if ( (_BYTE)a2 && (v4 = a2, sub_22416F0((__int64 *)a1, "-a", 0, 2u) == -1) )
  {
    if ( 0x3FFFFFFFFFFFFFFFLL - a1[1] <= 5 )
      goto LABEL_11;
    sub_2241490(a1, "-a:8:8", 6u);
  }
  else
  {
    v4 = 0;
  }
  if ( (_BYTE)a3 && sub_22416F0((__int64 *)a1, "i128", 0, 4u) == -1 )
  {
    if ( 0x3FFFFFFFFFFFFFFFLL - a1[1] > 0xC )
    {
      v4 = a3;
      sub_2241490(a1, "-i128:128:128", 0xDu);
      return v4;
    }
LABEL_11:
    sub_4262D8((__int64)"basic_string::append");
  }
  return v4;
}
