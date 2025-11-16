// Function: sub_BB5510
// Address: 0xbb5510
//
char __fastcall sub_BB5510(unsigned __int8 *a1)
{
  char result; // al

  result = sub_BB5300((__int64)a1);
  if ( !result && *a1 > 0x1Cu )
  {
    result = sub_B44AB0(a1);
    if ( !result )
      return sub_B44930((__int64)a1);
  }
  return result;
}
