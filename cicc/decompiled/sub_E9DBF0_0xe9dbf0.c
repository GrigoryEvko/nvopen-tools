// Function: sub_E9DBF0
// Address: 0xe9dbf0
//
__int64 __fastcall sub_E9DBF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // [rsp+0h] [rbp-90h] BYREF
  int v8; // [rsp+8h] [rbp-88h]
  __int64 v9; // [rsp+10h] [rbp-80h]
  char v10; // [rsp+20h] [rbp-70h]
  __int64 v11; // [rsp+28h] [rbp-68h]
  __int64 v12; // [rsp+30h] [rbp-60h]
  __int64 v13; // [rsp+38h] [rbp-58h]
  __int64 v14; // [rsp+40h] [rbp-50h]
  _QWORD *v15; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v16[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v5 = 1;
  if ( v4 != sub_E97650 )
    v5 = v4();
  v11 = a3;
  v10 = 17;
  v7 = v5;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = v16;
  sub_E97AA0((__int64 *)&v15, byte_3F871B3, (__int64)byte_3F871B3);
  v9 = a2;
  v8 = 0;
  result = sub_E99320(a1);
  if ( result )
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v7);
  if ( v15 != v16 )
    result = j_j___libc_free_0(v15, v16[0] + 1LL);
  if ( v12 )
    return j_j___libc_free_0(v12, v14 - v12);
  return result;
}
