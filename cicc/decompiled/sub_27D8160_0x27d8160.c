// Function: sub_27D8160
// Address: 0x27d8160
//
__int64 __fastcall sub_27D8160(__int64 *a1, unsigned __int8 *a2, unsigned int a3, unsigned int a4)
{
  unsigned int v4; // r12d
  unsigned int v6; // eax

  v4 = a3;
  if ( (unsigned __int8)a3 >= (unsigned __int8)a4 )
    return a3;
  v6 = sub_F51790(a2, a4, *a1);
  if ( (unsigned __int8)v4 < (unsigned __int8)v6 )
    return v6;
  return v4;
}
