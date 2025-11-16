// Function: sub_327FD70
// Address: 0x327fd70
//
__int64 __fastcall sub_327FD70(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // dx
  __int64 v6; // r8
  __int64 v7; // r9

  if ( BYTE4(a4) )
  {
    v5 = sub_2D43AD0(a2, a4);
    if ( v5 )
      return v5;
  }
  else
  {
    v5 = sub_2D43050(a2, a4);
    if ( v5 )
      return v5;
  }
  return sub_3009450(a1, a2, a3, a4, v6, v7);
}
