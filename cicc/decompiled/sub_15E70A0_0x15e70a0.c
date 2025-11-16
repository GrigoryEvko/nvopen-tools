// Function: sub_15E70A0
// Address: 0x15e70a0
//
__int64 __fastcall sub_15E70A0(__int64 a1, char *a2, signed __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 *v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v12; // [rsp+8h] [rbp-38h]

  v7 = (__int64 *)sub_15996B0(*(_QWORD *)(a1 + 24), a2, a3, 1);
  v12 = *v7;
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL) + 40LL);
  v9 = sub_1648A60(88, 1);
  v10 = v9;
  if ( v9 )
    sub_15E51E0(v9, v8, v12, 1, 8, (__int64)v7, a4, 0, 0, a5, 0);
  *(_BYTE *)(v10 + 32) = *(_BYTE *)(v10 + 32) & 0x3F | 0x80;
  return v10;
}
