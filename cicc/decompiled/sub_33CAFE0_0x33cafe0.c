// Function: sub_33CAFE0
// Address: 0x33cafe0
//
__int64 __fastcall sub_33CAFE0(int a1)
{
  __int64 v1; // rdi

  v1 = (unsigned int)(a1 - 180);
  if ( (unsigned int)v1 > 3 )
    BUG();
  return dword_44DF820[v1];
}
