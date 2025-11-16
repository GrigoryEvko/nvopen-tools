// Function: sub_DA51B0
// Address: 0xda51b0
//
void __fastcall sub_DA51B0(unsigned __int64 *a1, unsigned __int64 *a2, _QWORD **a3)
{
  __int64 v4; // rcx
  unsigned __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_DA5080(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = &a1[v4];
    v6 = (8 * v4) >> 3;
    sub_DA51B0(a1, v5);
    sub_DA51B0(v5, a2);
    sub_DA4D40(a1, v5, (__int64)a2, v6, a2 - v5, (__int64)a3);
  }
}
