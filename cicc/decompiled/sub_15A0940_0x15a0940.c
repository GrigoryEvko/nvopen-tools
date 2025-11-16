// Function: sub_15A0940
// Address: 0x15a0940
//
__int64 __fastcall sub_15A0940(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rdi
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // r12
  _BYTE v13[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v14; // [rsp+8h] [rbp-38h] BYREF
  __int64 v15; // [rsp+10h] [rbp-30h]

  if ( *(_BYTE *)(sub_1595890(a1) + 8) != 1
    && *(_BYTE *)(sub_1595890(a1) + 8) != 2
    && *(_BYTE *)(sub_1595890(a1) + 8) != 3 )
  {
    v8 = sub_1595A50(a1, a2);
    v9 = sub_1595890(a1);
    return sub_15A0680(v9, v8, 0);
  }
  sub_1595B70((__int64)v13, a1, a2);
  v2 = (_QWORD *)sub_16498A0(a1);
  v3 = sub_159CCF0(v2, (__int64)v13);
  v6 = sub_16982C0(v2, v13, v4, v5);
  if ( v14 != v6 )
  {
    sub_1698460(&v14);
    return v3;
  }
  v10 = v15;
  if ( !v15 )
    return v3;
  v11 = 32LL * *(_QWORD *)(v15 - 8);
  v12 = v15 + v11;
  if ( v15 != v15 + v11 )
  {
    do
    {
      v12 -= 32;
      sub_127D120((_QWORD *)(v12 + 8));
    }
    while ( v10 != v12 );
  }
  j_j_j___libc_free_0_0(v10 - 8);
  return v3;
}
