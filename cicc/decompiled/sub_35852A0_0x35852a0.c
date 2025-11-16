// Function: sub_35852A0
// Address: 0x35852a0
//
__int64 sub_35852A0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  volatile signed __int32 *v2; // rdi
  volatile signed __int32 *v4; // [rsp+8h] [rbp-78h] BYREF
  __int64 v5[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v7[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+40h] [rbp-40h] BYREF

  v5[0] = (__int64)v6;
  sub_3583D30(v5, byte_3F871B3, (__int64)byte_3F871B3);
  v7[0] = (__int64)v8;
  sub_3583D30(v7, byte_3F871B3, (__int64)byte_3F871B3);
  v4 = 0;
  v0 = sub_22077B0(0x110u);
  v1 = v0;
  if ( v0 )
    sub_3584CF0(v0, (__int64)v5, v7, 1, &v4);
  v2 = v4;
  if ( v4 && !_InterlockedSub(v4 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  if ( (_QWORD *)v7[0] != v8 )
    j_j___libc_free_0(v7[0]);
  if ( (_QWORD *)v5[0] != v6 )
    j_j___libc_free_0(v5[0]);
  return v1;
}
