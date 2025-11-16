// Function: sub_1E794C0
// Address: 0x1e794c0
//
void __fastcall sub_1E794C0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_1E79020(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = &a1[v4];
    v6 = (8 * v4) >> 3;
    sub_1E794C0(a1, v5);
    sub_1E794C0(v5, a2);
    sub_1E792E0(a1, v5, (__int64)a2, v6, a2 - v5, a3);
  }
}
