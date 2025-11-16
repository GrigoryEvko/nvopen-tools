// Function: sub_1E39D10
// Address: 0x1e39d10
//
__int64 __fastcall sub_1E39D10(char *a1, unsigned __int64 *a2, char *a3, __int64 a4)
{
  __int64 v6; // r9
  char *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 16 * (((((char *)a2 - a1) >> 4) + 1) / 2);
  v10 = v6;
  v7 = &a1[v6];
  if ( ((((char *)a2 - a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_1E38910(a1, (unsigned __int64 *)&a1[v6], a3);
    sub_1E38910(v7, a2, a3);
  }
  else
  {
    sub_1E39D10(a1, &a1[v6], a3);
    sub_1E39D10(v7, a2, a3);
  }
  sub_1E38BA0(a1, (unsigned int **)v7, (__int64)a2, v10 >> 4, ((char *)a2 - v7) >> 4, a3, a4);
  return v9;
}
