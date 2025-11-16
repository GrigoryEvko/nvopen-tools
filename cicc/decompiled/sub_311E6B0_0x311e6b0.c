// Function: sub_311E6B0
// Address: 0x311e6b0
//
void __fastcall sub_311E6B0(unsigned __int64 **a1, unsigned __int64 **a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_311E330(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = (char *)&a1[v4];
    v6 = (8 * v4) >> 3;
    sub_311E6B0(a1, v5);
    sub_311E6B0(v5, a2);
    sub_311E520(a1, v5, (__int64)a2, v6, ((char *)a2 - v5) >> 3, a3);
  }
}
