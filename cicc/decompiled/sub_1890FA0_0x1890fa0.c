// Function: sub_1890FA0
// Address: 0x1890fa0
//
void __fastcall sub_1890FA0(char *a1, char *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 224 )
  {
    sub_1890640(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 5;
    v3 = &a1[16 * v2];
    v4 = (16 * v2) >> 4;
    sub_1890FA0(a1, v3);
    sub_1890FA0(v3, a2);
    sub_1890E40(a1, v3, (__int64)a2, v4, (a2 - v3) >> 4);
  }
}
