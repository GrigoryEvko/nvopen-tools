// Function: sub_E9D020
// Address: 0xe9d020
//
__int64 __fastcall sub_E9D020(_QWORD *a1, __int64 a2, const char *a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 (*v8)(void); // rax
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 result; // rax
  const char *v12; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-98h]
  __int16 v14; // [rsp+20h] [rbp-80h]
  __int64 v15; // [rsp+28h] [rbp-78h]
  __int64 v16; // [rsp+30h] [rbp-70h]
  __int64 v17; // [rsp+38h] [rbp-68h]
  __int64 v18; // [rsp+40h] [rbp-60h]
  _QWORD *v19; // [rsp+48h] [rbp-58h]
  __int64 v20; // [rsp+50h] [rbp-50h]
  _QWORD v21[9]; // [rsp+58h] [rbp-48h] BYREF

  v5 = 1;
  v8 = *(__int64 (**)(void))(*a1 + 88LL);
  if ( v8 != sub_E97650 )
    v5 = v8();
  v12 = a3;
  v9 = a1[1];
  v13 = a4;
  v14 = 261;
  v10 = sub_E6C460(v9, &v12);
  result = sub_E99320((__int64)a1);
  if ( result )
  {
    LOBYTE(v14) = 18;
    v12 = (const char *)v5;
    v15 = a2;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = v21;
    v20 = 0;
    LOBYTE(v21[0]) = 0;
    v13 = v10;
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v12);
    if ( v19 != v21 )
      result = j_j___libc_free_0(v19, v21[0] + 1LL);
    if ( v16 )
      return j_j___libc_free_0(v16, v18 - v16);
  }
  return result;
}
