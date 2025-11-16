// Function: sub_E9D130
// Address: 0xe9d130
//
__int64 __fastcall sub_E9D130(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v10; // [rsp+0h] [rbp-A0h] BYREF
  int v11; // [rsp+8h] [rbp-98h]
  __int64 v12; // [rsp+10h] [rbp-90h]
  char v13; // [rsp+20h] [rbp-80h]
  __int64 v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h]
  __int64 v16; // [rsp+38h] [rbp-68h]
  __int64 v17; // [rsp+40h] [rbp-60h]
  _QWORD *v18; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v19[9]; // [rsp+58h] [rbp-48h] BYREF

  v6 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v7 = 1;
  if ( v6 != sub_E97650 )
    v7 = v6();
  v14 = a4;
  v13 = 7;
  v10 = v7;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = v19;
  sub_E97AA0((__int64 *)&v18, byte_3F871B3, (__int64)byte_3F871B3);
  v12 = a3;
  v11 = a2;
  result = sub_E99320(a1);
  v9 = result;
  if ( result )
  {
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v10);
    *(_DWORD *)(v9 + 56) = a2;
  }
  if ( v18 != v19 )
    result = j_j___libc_free_0(v18, v19[0] + 1LL);
  if ( v15 )
    return j_j___libc_free_0(v15, v17 - v15);
  return result;
}
