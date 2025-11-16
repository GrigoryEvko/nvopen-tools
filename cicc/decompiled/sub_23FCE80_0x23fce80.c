// Function: sub_23FCE80
// Address: 0x23fce80
//
void __fastcall sub_23FCE80(__int64 ***a1, __int64 ***a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 ***v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_23FB520(a1, a2);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = &a1[v4];
    v6 = (8 * v4) >> 3;
    sub_23FCE80(a1, v5);
    sub_23FCE80(v5, a2);
    sub_23FCCD0(a1, v5, (__int64)a2, v6, a2 - v5, a3);
  }
}
