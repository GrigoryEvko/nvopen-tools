// Function: sub_EFFA20
// Address: 0xeffa20
//
__int64 __fastcall sub_EFFA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ecx
  __int64 result; // rax
  __int64 v9; // [rsp-10h] [rbp-B0h]
  unsigned int v10; // [rsp+8h] [rbp-98h]
  _QWORD v11[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v13[6]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v14; // [rsp+60h] [rbp-40h]

  v6 = 0;
  v7 = *(_DWORD *)(a1 + 1060);
  *(_DWORD *)(a1 + 1056) = 0;
  if ( !v7 )
  {
    sub_C8D5F0(a1 + 1048, (const void *)(a1 + 1064), 1u, 8u, a5, a6);
    v6 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v6) = 3;
  ++*(_DWORD *)(a1 + 1056);
  v13[5] = 0x100000000LL;
  v14 = v11;
  v11[0] = v12;
  v11[1] = 0;
  LOBYTE(v12[0]) = 0;
  memset(&v13[1], 0, 32);
  v13[0] = &unk_49DD210;
  sub_CB5980((__int64)v13, 0, 0, 0);
  sub_F02A30(a2, v13);
  sub_EFE900(
    a1 + 1576,
    *(_QWORD *)(a1 + 1752),
    *(_QWORD *)(a1 + 1048),
    *(unsigned int *)(a1 + 1056),
    *v14,
    v14[1],
    v10,
    0);
  v13[0] = &unk_49DD210;
  sub_CB5840((__int64)v13);
  result = v9;
  if ( (_QWORD *)v11[0] != v12 )
    return j_j___libc_free_0(v11[0], v12[0] + 1LL);
  return result;
}
