// Function: sub_1633F60
// Address: 0x1633f60
//
__int64 __fastcall sub_1633F60(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v13[2]; // [rsp+8h] [rbp-100h] BYREF
  __int64 v14; // [rsp+18h] [rbp-F0h]
  _QWORD v15[2]; // [rsp+28h] [rbp-E0h] BYREF
  __int64 v16; // [rsp+38h] [rbp-D0h]
  __int64 v17[2]; // [rsp+48h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+58h] [rbp-B0h]
  __int64 v19[2]; // [rsp+68h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+78h] [rbp-90h]
  __int64 v21; // [rsp+88h] [rbp-80h] BYREF
  __int64 v22; // [rsp+90h] [rbp-78h]
  __int64 v23; // [rsp+98h] [rbp-70h]
  __int64 v24; // [rsp+A8h] [rbp-60h] BYREF
  __int64 v25; // [rsp+B0h] [rbp-58h]
  __int64 v26; // [rsp+B8h] [rbp-50h]
  _QWORD v27[2]; // [rsp+C8h] [rbp-40h] BYREF
  __int64 v28; // [rsp+D8h] [rbp-30h]

  v3 = *a2;
  v24 = 0;
  *a2 = 0;
  v27[0] = v3;
  v4 = a2[1];
  v25 = 0;
  v27[1] = v4;
  v5 = a2[2];
  a2[1] = 0;
  v28 = v5;
  a2[2] = 0;
  v26 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v19[0] = 0;
  v19[1] = 0;
  v20 = 0;
  v17[0] = 0;
  v17[1] = 0;
  v18 = 0;
  v15[0] = 0;
  v15[1] = 0;
  v16 = 0;
  v13[0] = 0;
  v13[1] = 0;
  v14 = 0;
  sub_142CF20(a1, 49, 0, 0, v13, v27, v15, v17, v19, &v21, &v24);
  if ( v13[0] )
    j_j___libc_free_0(v13[0], v14 - v13[0]);
  if ( v27[0] )
    j_j___libc_free_0(v27[0], v28 - v27[0]);
  if ( v15[0] )
    j_j___libc_free_0(v15[0], v16 - v15[0]);
  if ( v17[0] )
    j_j___libc_free_0(v17[0], v18 - v17[0]);
  if ( v19[0] )
    j_j___libc_free_0(v19[0], v20 - v19[0]);
  v6 = v22;
  v7 = v21;
  if ( v22 != v21 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 16);
      if ( v8 )
        j_j___libc_free_0(v8, *(_QWORD *)(v7 + 32) - v8);
      v7 += 40;
    }
    while ( v6 != v7 );
    v7 = v21;
  }
  if ( v7 )
    j_j___libc_free_0(v7, v23 - v7);
  v9 = v25;
  v10 = v24;
  if ( v25 != v24 )
  {
    do
    {
      v11 = *(_QWORD *)(v10 + 16);
      if ( v11 )
        j_j___libc_free_0(v11, *(_QWORD *)(v10 + 32) - v11);
      v10 += 40;
    }
    while ( v9 != v10 );
    v10 = v24;
  }
  if ( v10 )
    j_j___libc_free_0(v10, v26 - v10);
  return a1;
}
