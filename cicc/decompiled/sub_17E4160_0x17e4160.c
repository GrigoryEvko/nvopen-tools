// Function: sub_17E4160
// Address: 0x17e4160
//
void __fastcall sub_17E4160(char *a1, char *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 112 )
  {
    sub_17E25A0(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 4;
    v3 = (__int64)&a1[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_17E4160(a1, v3);
    sub_17E4160(v3, a2);
    sub_17E4010((__int64)a1, v3, (__int64)a2, v4, (__int64)&a2[-v3] >> 3);
  }
}
