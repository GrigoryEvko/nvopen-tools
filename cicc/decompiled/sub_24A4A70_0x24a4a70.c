// Function: sub_24A4A70
// Address: 0x24a4a70
//
void __fastcall sub_24A4A70(unsigned __int64 *a1, unsigned __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_24A3210(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = (__int64)&a1[v2];
    v4 = (8 * v2) >> 3;
    sub_24A4A70(a1, v3);
    sub_24A4A70(v3, a2);
    sub_24A4920((__int64)a1, v3, (__int64)a2, v4, ((__int64)a2 - v3) >> 3);
  }
}
