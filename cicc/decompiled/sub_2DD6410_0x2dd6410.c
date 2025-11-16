// Function: sub_2DD6410
// Address: 0x2dd6410
//
void __fastcall sub_2DD6410(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_2DD5F00(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = (char *)&a1[v4];
    v6 = (8 * v4) >> 3;
    sub_2DD6410(a1, v5);
    sub_2DD6410(v5, a2);
    sub_2DD6200((char *)a1, v5, (__int64)a2, v6, ((char *)a2 - v5) >> 3, a3);
  }
}
