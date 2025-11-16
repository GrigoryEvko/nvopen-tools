// Function: sub_F084D0
// Address: 0xf084d0
//
void __fastcall sub_F084D0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_F07380(a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)a1) >> 4;
    v3 = &a1[v2];
    v4 = (8 * v2) >> 3;
    sub_F084D0(a1, v3);
    sub_F084D0(v3, a2);
    sub_F08330(a1, v3, (__int64)a2, v4, a2 - v3);
  }
}
