// Function: sub_3585380
// Address: 0x3585380
//
__int64 __fastcall sub_3585380(__int64 a1, __int64 a2, int a3, volatile signed __int32 **a4)
{
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  volatile signed __int32 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  volatile signed __int32 *v14; // rdi
  volatile signed __int32 *v16; // [rsp+18h] [rbp-78h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v19[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+50h] [rbp-40h] BYREF

  v7 = *(_BYTE **)a1;
  v8 = *(_QWORD *)(a1 + 8);
  v17[0] = (__int64)v18;
  sub_3583BC0(v17, v7, (__int64)&v7[v8]);
  v9 = *(_BYTE **)a2;
  v10 = *(_QWORD *)a2 + *(_QWORD *)(a2 + 8);
  v19[0] = (__int64)v20;
  sub_3583BC0(v19, v9, v10);
  v11 = *a4;
  *a4 = 0;
  v16 = v11;
  v12 = sub_22077B0(0x110u);
  v13 = v12;
  if ( v12 )
    sub_3584CF0(v12, (__int64)v17, v19, a3, &v16);
  v14 = v16;
  if ( v16 && !_InterlockedSub(v16 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 8LL))(v14);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  return v13;
}
