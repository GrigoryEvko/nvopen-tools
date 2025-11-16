// Function: sub_2427D20
// Address: 0x2427d20
//
void __fastcall sub_2427D20(unsigned __int64 *a1, unsigned __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_2425AC0(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = (__int64)&a1[v2];
    v4 = (8 * v2) >> 3;
    sub_2427D20(a1, v3);
    sub_2427D20(v3, a2);
    sub_2427BD0((__int64)a1, v3, (__int64)a2, v4, ((__int64)a2 - v3) >> 3);
  }
}
