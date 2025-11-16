// Function: sub_266B3D0
// Address: 0x266b3d0
//
__int64 __fastcall sub_266B3D0(char *a1, char *a2, char *a3, __int64 a4)
{
  __int64 v6; // r9
  char *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = 16 * ((((a2 - a1) >> 4) + 1) / 2);
  v10 = v6;
  v7 = &a1[v6];
  if ( (((a2 - a1) >> 4) + 1) / 2 <= a4 )
  {
    sub_2664D00(a1, &a1[v6], a3);
    sub_2664D00(v7, a2, a3);
  }
  else
  {
    sub_266B3D0(a1, &a1[v6], a3);
    sub_266B3D0(v7, a2, a3);
  }
  sub_266B040(a1, (unsigned __int64 *)v7, (__int64)a2, v10 >> 4, (a2 - v7) >> 4, a3, (char *)a4);
  return v9;
}
