// Function: sub_3724720
// Address: 0x3724720
//
__int64 __fastcall sub_3724720(__int64 *a1, __int64 *a2, char *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - a1 + 1) / 2;
  v10 = 8 * v6;
  v7 = &a1[v6];
  if ( v6 <= a4 )
  {
    sub_3722970(a1, &a1[v6], a3);
    sub_3722970(v7, a2, a3);
  }
  else
  {
    sub_3724720(a1, &a1[v6], a3);
    sub_3724720(v7, a2, a3);
  }
  sub_3724400((char *)a1, (char *)v7, (__int64)a2, v10 >> 3, a2 - v7, (void **)a3, a4);
  return v9;
}
