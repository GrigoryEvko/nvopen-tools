// Function: sub_2D92D10
// Address: 0x2d92d10
//
__int64 *__fastcall sub_2D92D10(__int64 *a1)
{
  int v1; // ebx
  char *v2; // rax
  __int64 v3; // rdx
  __int64 v5[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v6; // [rsp+10h] [rbp-30h] BYREF

  sub_2D90F60(v5);
  v1 = sub_2241AC0((__int64)v5, "native");
  if ( (__int64 *)v5[0] != &v6 )
    j_j___libc_free_0(v5[0]);
  if ( v1 )
  {
    sub_2D90F60(a1);
  }
  else
  {
    v2 = sub_12571D0();
    *a1 = (__int64)(a1 + 2);
    sub_2D8E390(a1, v2, (__int64)&v2[v3]);
  }
  return a1;
}
