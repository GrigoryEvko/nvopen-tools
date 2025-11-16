// Function: sub_1D95450
// Address: 0x1d95450
//
void __fastcall sub_1D95450(char *a1, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( a2 - a1 <= 112 )
  {
    sub_1D93DF0(a1, a2);
  }
  else
  {
    v4 = (a2 - a1) >> 4;
    v5 = (__int64 *)&a1[8 * v4];
    v6 = (8 * v4) >> 3;
    sub_1D95450(a1, v5);
    sub_1D95450(v5, a2);
    sub_1D95270((__int64 *)a1, v5, (__int64)a2, v6, (a2 - (char *)v5) >> 3, a3);
  }
}
