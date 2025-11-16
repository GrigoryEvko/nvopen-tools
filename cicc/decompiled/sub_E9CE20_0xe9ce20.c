// Function: sub_E9CE20
// Address: 0xe9ce20
//
__int64 __fastcall sub_E9CE20(__int64 a1, int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 (*v8)(void); // rdx
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rbx
  __int64 v12; // [rsp+10h] [rbp-A0h] BYREF
  int v13; // [rsp+18h] [rbp-98h]
  __int64 v14; // [rsp+20h] [rbp-90h]
  int v15; // [rsp+28h] [rbp-88h]
  char v16; // [rsp+30h] [rbp-80h]
  __int64 v17; // [rsp+38h] [rbp-78h]
  __int64 v18; // [rsp+40h] [rbp-70h]
  __int64 v19; // [rsp+48h] [rbp-68h]
  __int64 v20; // [rsp+50h] [rbp-60h]
  _QWORD *v21; // [rsp+58h] [rbp-58h]
  __int64 v22; // [rsp+60h] [rbp-50h]
  _QWORD v23[9]; // [rsp+68h] [rbp-48h] BYREF

  v8 = *(__int64 (**)(void))(*(_QWORD *)a1 + 88LL);
  v9 = 1;
  if ( v8 != sub_E97650 )
    v9 = v8();
  v17 = a5;
  v15 = a4;
  v12 = v9;
  v16 = 4;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = v23;
  v22 = 0;
  LOBYTE(v23[0]) = 0;
  v13 = a2;
  v14 = a3;
  result = sub_E99320(a1);
  v11 = result;
  if ( result )
  {
    result = sub_E9CD20((__int64 *)(result + 32), (__int64)&v12);
    *(_DWORD *)(v11 + 56) = a2;
  }
  if ( v21 != v23 )
    result = j_j___libc_free_0(v21, v23[0] + 1LL);
  if ( v18 )
    return j_j___libc_free_0(v18, v20 - v18);
  return result;
}
