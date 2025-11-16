// Function: sub_E9CF30
// Address: 0xe9cf30
//
__int64 __fastcall sub_E9CF30(__int64 a1, int a2, int a3, __int64 a4)
{
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // [rsp+10h] [rbp-90h] BYREF
  int v10; // [rsp+18h] [rbp-88h]
  int v11; // [rsp+1Ch] [rbp-84h]
  char v12; // [rsp+30h] [rbp-70h]
  __int64 v13; // [rsp+38h] [rbp-68h]
  __int64 v14; // [rsp+40h] [rbp-60h]
  __int64 v15; // [rsp+48h] [rbp-58h]
  __int64 v16; // [rsp+50h] [rbp-50h]
  _QWORD *v17; // [rsp+58h] [rbp-48h]
  __int64 v18; // [rsp+60h] [rbp-40h]
  _QWORD v19[7]; // [rsp+68h] [rbp-38h] BYREF

  v6 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v7 = 1;
  if ( v6 != sub_E97650 )
    v7 = v6();
  v13 = a4;
  v9 = v7;
  v12 = 13;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = v19;
  v18 = 0;
  LOBYTE(v19[0]) = 0;
  v10 = a2;
  v11 = a3;
  result = sub_E99320(a1);
  if ( result )
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v9);
  if ( v17 != v19 )
    result = j_j___libc_free_0(v17, v19[0] + 1LL);
  if ( v14 )
    return j_j___libc_free_0(v14, v16 - v14);
  return result;
}
