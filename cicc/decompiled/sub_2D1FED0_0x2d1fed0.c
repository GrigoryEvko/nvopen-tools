// Function: sub_2D1FED0
// Address: 0x2d1fed0
//
__int64 __fastcall sub_2D1FED0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  char *v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  sub_2C75F20((__int64)v9, a7);
  sub_2241490((unsigned __int64 *)a1, v9[0], (size_t)v9[1]);
  if ( (__int64 *)v9[0] != &v10 )
    j_j___libc_free_0((unsigned __int64)v9[0]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 9 )
    goto LABEL_7;
  sub_2241490((unsigned __int64 *)a1, " : Error: ", 0xAu);
  if ( a3 != 1 )
    BUG();
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 0x33 )
LABEL_7:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)a1, "a function that is not __global__ cannot be launched", 0x34u);
  return a1;
}
