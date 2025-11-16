// Function: sub_161DEF0
// Address: 0x161def0
//
void __fastcall sub_161DEF0(char *a1, unsigned int *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - a1 <= 224 )
  {
    sub_161D9B0((__int64)a1, a2);
  }
  else
  {
    v2 = ((char *)a2 - a1) >> 5;
    v3 = &a1[16 * v2];
    v4 = (16 * v2) >> 4;
    sub_161DEF0(a1, v3);
    sub_161DEF0(v3, a2);
    sub_161DD90(a1, v3, (__int64)a2, v4, ((char *)a2 - v3) >> 4);
  }
}
