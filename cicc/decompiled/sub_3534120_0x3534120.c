// Function: sub_3534120
// Address: 0x3534120
//
__int64 __fastcall sub_3534120(char *a1, char *a2, char *a3, __int64 a4)
{
  __int64 v6; // rax
  char *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (((a2 - a1) >> 3) + 1) / 2;
  v10 = 8 * v6;
  v7 = &a1[8 * v6];
  if ( v6 <= a4 )
  {
    sub_3533680(a1, &a1[8 * v6], a3);
    sub_3533680(v7, a2, a3);
  }
  else
  {
    sub_3534120(a1, &a1[8 * v6], a3);
    sub_3534120(v7, a2, a3);
  }
  sub_3533890(a1, v7, (__int64)a2, v10 >> 3, (a2 - v7) >> 3, a3, a4);
  return v9;
}
