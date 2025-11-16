// Function: sub_E9D410
// Address: 0xe9d410
//
__int64 __fastcall sub_E9D410(__int64 a1, int a2, __int64 a3)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // [rsp+0h] [rbp-A0h] BYREF
  int v9; // [rsp+8h] [rbp-98h]
  __int64 v10; // [rsp+10h] [rbp-90h]
  char v11; // [rsp+20h] [rbp-80h]
  __int64 v12; // [rsp+28h] [rbp-78h]
  __int64 v13; // [rsp+30h] [rbp-70h]
  __int64 v14; // [rsp+38h] [rbp-68h]
  __int64 v15; // [rsp+40h] [rbp-60h]
  _QWORD *v16; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v17[9]; // [rsp+58h] [rbp-48h] BYREF

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v5 = 1;
  if ( v4 != sub_E97650 )
    v5 = v4();
  v12 = a3;
  v11 = 5;
  v8 = v5;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = v17;
  sub_E97AA0((__int64 *)&v16, byte_3F871B3, (__int64)byte_3F871B3);
  v9 = a2;
  v10 = 0;
  result = sub_E99320(a1);
  v7 = result;
  if ( result )
  {
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v8);
    *(_DWORD *)(v7 + 56) = a2;
  }
  if ( v16 != v17 )
    result = j_j___libc_free_0(v16, v17[0] + 1LL);
  if ( v13 )
    return j_j___libc_free_0(v13, v15 - v13);
  return result;
}
