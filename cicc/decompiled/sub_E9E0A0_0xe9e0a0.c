// Function: sub_E9E0A0
// Address: 0xe9e0a0
//
__int64 __fastcall sub_E9E0A0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // [rsp+0h] [rbp-A0h] BYREF
  int v10; // [rsp+8h] [rbp-98h]
  __int64 v11; // [rsp+10h] [rbp-90h]
  char v12; // [rsp+20h] [rbp-80h]
  __int64 v13; // [rsp+28h] [rbp-78h]
  __int64 v14; // [rsp+30h] [rbp-70h]
  __int64 v15; // [rsp+38h] [rbp-68h]
  __int64 v16; // [rsp+40h] [rbp-60h]
  _QWORD *v17; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v18[9]; // [rsp+58h] [rbp-48h] BYREF

  v6 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v7 = 1;
  if ( v6 != sub_E97650 )
    v7 = v6();
  v13 = a4;
  v12 = 19;
  v9 = v7;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = v18;
  sub_E97AA0((__int64 *)&v17, byte_3F871B3, (__int64)byte_3F871B3);
  v10 = a2;
  v11 = a3;
  result = sub_E99320(a1);
  if ( result )
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v9);
  if ( v17 != v18 )
    result = j_j___libc_free_0(v17, v18[0] + 1LL);
  if ( v14 )
    return j_j___libc_free_0(v14, v16 - v14);
  return result;
}
