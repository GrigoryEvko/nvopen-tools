// Function: sub_2F51120
// Address: 0x2f51120
//
__int64 __fastcall sub_2F51120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  size_t v7; // r13
  size_t v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r8
  unsigned int v11; // r12d
  char v15; // [rsp+28h] [rbp-48h]
  int v17[2]; // [rsp+38h] [rbp-38h] BYREF

  v15 = byte_4F826E9[0];
  v7 = strlen("Register Allocation");
  v8 = strlen("regalloc");
  sub_CA08F0((__int64 *)v17, "evict", 5u, (__int64)"Evict", 5, v15, "regalloc", v8, "Register Allocation", v7);
  v9 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 968) + 16LL))(
         *(_QWORD *)(a1 + 968),
         a2,
         a3,
         a5,
         a6);
  v11 = v9;
  if ( v9 )
    sub_2F50AD0(a1, a2, v9, a4, v10);
  if ( *(_QWORD *)v17 )
    sub_C9E2A0(*(__int64 *)v17);
  return v11;
}
