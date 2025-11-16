// Function: sub_17E3F90
// Address: 0x17e3f90
//
void __fastcall sub_17E3F90(char *a1, char *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 112 )
  {
    sub_17E26D0(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 4;
    v3 = (__int64)&a1[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_17E3F90(a1, v3);
    sub_17E3F90(v3, a2);
    sub_17E3E40((__int64)a1, v3, (__int64)a2, v4, (__int64)&a2[-v3] >> 3);
  }
}
