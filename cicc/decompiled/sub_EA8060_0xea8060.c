// Function: sub_EA8060
// Address: 0xea8060
//
__int64 __fastcall sub_EA8060(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r14
  _QWORD *v7; // r15
  bool v8; // bl
  _QWORD *v10; // rbx
  _QWORD *v11; // r14
  __int64 *v12; // rdi
  bool v13; // [rsp+7h] [rbp-179h]
  _QWORD v17[4]; // [rsp+50h] [rbp-130h] BYREF
  _QWORD *v18; // [rsp+70h] [rbp-110h]
  _QWORD v19[2]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD *v20; // [rsp+90h] [rbp-F0h]
  _QWORD v21[2]; // [rsp+A0h] [rbp-E0h] BYREF
  _QWORD *v22; // [rsp+B0h] [rbp-D0h]
  _QWORD v23[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD *v24; // [rsp+D0h] [rbp-B0h]
  _QWORD v25[2]; // [rsp+E0h] [rbp-A0h] BYREF
  _QWORD *v26; // [rsp+F0h] [rbp-90h]
  _QWORD v27[2]; // [rsp+100h] [rbp-80h] BYREF
  _QWORD *v28; // [rsp+110h] [rbp-70h]
  _QWORD v29[2]; // [rsp+120h] [rbp-60h] BYREF
  _QWORD *v30; // [rsp+130h] [rbp-50h]
  _QWORD *v31; // [rsp+138h] [rbp-48h]
  __int64 v32; // [rsp+140h] [rbp-40h]

  sub_EA7E40((__int64)v17, (int *)(a1[1] + 80LL));
  v6 = v31;
  v7 = v30;
  v8 = (v17[0] & 8) != 0;
  if ( v31 != v30 )
  {
    do
    {
      if ( (_QWORD *)*v7 != v7 + 2 )
        j_j___libc_free_0(*v7, v7[2] + 1LL);
      v7 += 4;
    }
    while ( v6 != v7 );
    v7 = v30;
  }
  if ( v7 )
    j_j___libc_free_0(v7, v32 - (_QWORD)v7);
  if ( v28 != v29 )
    j_j___libc_free_0(v28, v29[0] + 1LL);
  if ( v26 != v27 )
    j_j___libc_free_0(v26, v27[0] + 1LL);
  if ( v24 != v25 )
    j_j___libc_free_0(v24, v25[0] + 1LL);
  if ( v22 != v23 )
    j_j___libc_free_0(v22, v23[0] + 1LL);
  if ( v20 != v21 )
    j_j___libc_free_0(v20, v21[0] + 1LL);
  if ( v18 != v19 )
    j_j___libc_free_0(v18, v19[0] + 1LL);
  if ( v8 )
    return 0;
  sub_EA7E40((__int64)v17, (int *)(a1[1] + 80LL));
  v10 = v31;
  v11 = v30;
  v13 = (v17[0] & 4) != 0;
  if ( v31 != v30 )
  {
    do
    {
      if ( (_QWORD *)*v11 != v11 + 2 )
        j_j___libc_free_0(*v11, v11[2] + 1LL);
      v11 += 4;
    }
    while ( v10 != v11 );
    v11 = v30;
  }
  if ( v11 )
    j_j___libc_free_0(v11, v32 - (_QWORD)v11);
  if ( v28 != v29 )
    j_j___libc_free_0(v28, v29[0] + 1LL);
  if ( v26 != v27 )
    j_j___libc_free_0(v26, v27[0] + 1LL);
  if ( v24 != v25 )
    j_j___libc_free_0(v24, v25[0] + 1LL);
  if ( v22 != v23 )
    j_j___libc_free_0(v22, v23[0] + 1LL);
  if ( v20 != v21 )
    j_j___libc_free_0(v20, v21[0] + 1LL);
  if ( v18 != v19 )
    j_j___libc_free_0(v18, v19[0] + 1LL);
  if ( !v13 )
  {
    v12 = (__int64 *)a1[31];
    v17[0] = a4;
    v17[1] = a5;
    sub_C91CB0(v12, a2, 1, a3, (__int64)v17, 1, 0, 0, 1u);
    sub_EA2AE0(a1);
    return 0;
  }
  return sub_ECDA70(a1, a2, a3, a4, a5);
}
