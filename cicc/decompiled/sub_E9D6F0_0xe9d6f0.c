// Function: sub_E9D6F0
// Address: 0xe9d6f0
//
__int64 __fastcall sub_E9D6F0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rdx
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // [rsp+0h] [rbp-90h] BYREF
  int v6; // [rsp+8h] [rbp-88h]
  __int64 v7; // [rsp+10h] [rbp-80h]
  char v8; // [rsp+20h] [rbp-70h]
  __int64 v9; // [rsp+28h] [rbp-68h]
  __int64 v10; // [rsp+30h] [rbp-60h]
  __int64 v11; // [rsp+38h] [rbp-58h]
  __int64 v12; // [rsp+40h] [rbp-50h]
  _QWORD *v13; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v14[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v3 = 1;
  if ( v2 != sub_E97650 )
    v3 = v2();
  v9 = a2;
  v8 = 1;
  v5 = v3;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = v14;
  sub_E97AA0((__int64 *)&v13, byte_3F871B3, (__int64)byte_3F871B3);
  v7 = 0;
  v6 = 0;
  result = sub_E99320(a1);
  if ( result )
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v5);
  if ( v13 != v14 )
    result = j_j___libc_free_0(v13, v14[0] + 1LL);
  if ( v10 )
    return j_j___libc_free_0(v10, v12 - v10);
  return result;
}
