// Function: sub_1E38B20
// Address: 0x1e38b20
//
void __fastcall sub_1E38B20(char *a1, unsigned __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - a1 <= 224 )
  {
    sub_1E38550((__int64)a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - a1) >> 5;
    v3 = (__int64)&a1[16 * v2];
    v4 = (16 * v2) >> 4;
    sub_1E38B20(a1, v3);
    sub_1E38B20(v3, a2);
    sub_1E37BE0(a1, v3, (__int64)a2, v4, ((__int64)a2 - v3) >> 4);
  }
}
