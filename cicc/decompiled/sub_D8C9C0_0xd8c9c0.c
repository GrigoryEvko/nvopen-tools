// Function: sub_D8C9C0
// Address: 0xd8c9c0
//
__int64 __fastcall sub_D8C9C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // [rsp-68h] [rbp-68h] BYREF
  __int64 v13; // [rsp-60h] [rbp-60h]
  __int64 v14; // [rsp-58h] [rbp-58h]
  unsigned int v15; // [rsp-50h] [rbp-50h]
  __int64 v16[2]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v17; // [rsp-38h] [rbp-38h] BYREF

  result = *(_QWORD *)(a1 + 40);
  if ( !result )
  {
    if ( !*(_QWORD *)(a1 + 24) )
      sub_4263D6(a1, a2, a3);
    v4 = (*(__int64 (__fastcall **)(__int64))(a1 + 32))(a1 + 8);
    v12 = *(_QWORD *)a1;
    v14 = v4;
    v13 = sub_B2BEC0(v12);
    v15 = sub_AE2980(v13, 0)[1];
    sub_AADB10((__int64)v16, v15, 1);
    v5 = sub_22077B0(104);
    v10 = v5;
    if ( v5 )
      sub_D8C2B0(v5, &v12, v6, v7, v8, v9);
    v11 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 40) = v10;
    if ( v11 )
    {
      sub_D85F30(*(_QWORD **)(v11 + 64));
      sub_D85E30(*(_QWORD **)(v11 + 16));
      j_j___libc_free_0(v11, 104);
    }
    sub_969240(&v17);
    sub_969240(v16);
    return *(_QWORD *)(a1 + 40);
  }
  return result;
}
