// Function: sub_B5B240
// Address: 0xb5b240
//
__int64 __fastcall sub_B5B240(int a1)
{
  __int64 v2; // [rsp+8h] [rbp-8h]

  if ( sub_B5B000(a1) )
  {
    LODWORD(v2) = 1;
    BYTE4(v2) = 1;
  }
  else
  {
    BYTE4(v2) = 0;
  }
  return v2;
}
