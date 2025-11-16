// Function: sub_1CEC1E0
// Address: 0x1cec1e0
//
__int64 __fastcall sub_1CEC1E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  const char *v8[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+10h] [rbp-20h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  sub_1C315E0((__int64)v8, a7);
  sub_2241490(a1, v8[0], v8[1]);
  if ( (__int64 *)v8[0] != &v9 )
    j_j___libc_free_0(v8[0], v9 + 1);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 9
    || (sub_2241490(a1, " : Error: ", 10), (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 0x33) )
  {
    sub_4262D8((__int64)"basic_string::append");
  }
  sub_2241490(a1, "a function that is not __global__ cannot be launched", 52);
  return a1;
}
