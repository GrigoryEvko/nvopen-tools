// Function: sub_3533600
// Address: 0x3533600
//
void __fastcall sub_3533600(char *a1, char *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 112 )
  {
    sub_3533450(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 4;
    v3 = &a1[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_3533600(a1, v3);
    sub_3533600(v3, a2);
    sub_3532EF0(a1, v3, (__int64)a2, v4, (a2 - v3) >> 3);
  }
}
