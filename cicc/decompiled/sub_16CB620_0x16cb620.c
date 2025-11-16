// Function: sub_16CB620
// Address: 0x16cb620
//
__int64 __fastcall sub_16CB620(unsigned __int64 a1, unsigned __int64 a2, char a3)
{
  unsigned int v3; // r8d

  v3 = -1;
  if ( a1 >> a3 >= a2 )
  {
    v3 = 1;
    if ( a1 >> a3 <= a2 )
      return a1 >> a3 << a3 < a1;
  }
  return v3;
}
