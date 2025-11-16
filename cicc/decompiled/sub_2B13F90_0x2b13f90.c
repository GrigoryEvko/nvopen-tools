// Function: sub_2B13F90
// Address: 0x2b13f90
//
void __fastcall sub_2B13F90(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 224 )
  {
    sub_2B0E6E0(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 5;
    v3 = a1 + 16 * v2;
    v4 = (16 * v2) >> 4;
    sub_2B13F90(a1, v3);
    sub_2B13F90(v3, a2);
    sub_2B13E30(a1, v3, a2, v4, (a2 - v3) >> 4);
  }
}
