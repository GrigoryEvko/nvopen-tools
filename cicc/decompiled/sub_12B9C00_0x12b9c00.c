// Function: sub_12B9C00
// Address: 0x12b9c00
//
__int64 __fastcall sub_12B9C00(_DWORD *a1)
{
  if ( !a1 )
    return 4;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    *a1 = 0;
    return 0;
  }
  else
  {
    *a1 = dword_4C6F008;
    return 0;
  }
}
