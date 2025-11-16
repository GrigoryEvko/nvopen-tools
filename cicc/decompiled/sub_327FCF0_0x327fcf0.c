// Function: sub_327FCF0
// Address: 0x327fcf0
//
__int64 __fastcall sub_327FCF0(__int64 *a1, unsigned int a2, __int64 a3, unsigned int a4, unsigned __int8 a5)
{
  unsigned __int16 v8; // dx

  if ( a5 )
  {
    v8 = sub_2D43AD0(a2, a4);
    if ( v8 )
      return v8;
  }
  else
  {
    v8 = sub_2D43050(a2, a4);
    if ( v8 )
      return v8;
  }
  return sub_3009400(a1, a2, a3, a4, a5);
}
