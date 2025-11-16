// Function: sub_DF2F80
// Address: 0xdf2f80
//
__int64 __fastcall sub_DF2F80(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // r12d
  _QWORD *v9; // rax
  __int64 result; // rax
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-68h]
  _BYTE v13[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14; // [rsp+20h] [rbp-50h]
  char v15; // [rsp+30h] [rbp-40h]

  v4 = sub_DEEF40(a1, a2);
  v8 = a3 & ~(unsigned int)sub_DC1810(v4, *(_QWORD *)(a1 + 112), v5, v6, v7);
  v9 = sub_DA4270(*(_QWORD *)(a1 + 112), v4, v8);
  sub_DEF380(a1, (__int64)v9);
  v11 = a2;
  v12 = v8;
  sub_D45B70((__int64)v13, a1 + 32, &v11);
  result = v14;
  if ( !v15 )
    *(_DWORD *)(v14 + 40) |= v8;
  return result;
}
