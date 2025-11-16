// Function: sub_30CD980
// Address: 0x30cd980
//
void __fastcall sub_30CD980(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rax
  char *v5; // r13
  size_t v6; // rax
  __int64 v7[2]; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-B0h] BYREF
  __int64 *v9; // [rsp+20h] [rbp-A0h]
  __int64 v10; // [rsp+30h] [rbp-90h] BYREF
  __int64 v11[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+70h] [rbp-50h]
  _QWORD v14[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( **(_BYTE **)a1 )
    sub_B18290(a2, " to match profiling context", 0x1Bu);
  sub_B18290(a2, " with ", 6u);
  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)v2 == 0x80000000 )
  {
    sub_B18290(a2, "(cost=always)", 0xDu);
  }
  else if ( *(_DWORD *)v2 == 0x7FFFFFFF )
  {
    sub_B18290(a2, "(cost=never)", 0xCu);
  }
  else
  {
    sub_B18290(a2, "(cost=", 6u);
    sub_B16530(v7, "Cost", 4, *(_DWORD *)v2);
    v3 = sub_23FD640(a2, (__int64)v7);
    sub_B18290(v3, ", threshold=", 0xCu);
    sub_B16530(v11, "Threshold", 9, *(_DWORD *)(v2 + 4));
    v4 = sub_23FD640(v3, (__int64)v11);
    sub_B18290(v4, ")", 1u);
    if ( v13 != v14 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( (_QWORD *)v11[0] != v12 )
      j_j___libc_free_0(v11[0]);
    if ( v9 != &v10 )
      j_j___libc_free_0((unsigned __int64)v9);
    if ( (__int64 *)v7[0] != &v8 )
      j_j___libc_free_0(v7[0]);
  }
  v5 = *(char **)(v2 + 16);
  if ( v5 )
  {
    sub_B18290(a2, ": ", 2u);
    v6 = strlen(v5);
    sub_B16430((__int64)v11, "Reason", 6u, v5, v6);
    sub_23FD640(a2, (__int64)v11);
    if ( v13 != v14 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( (_QWORD *)v11[0] != v12 )
      j_j___libc_free_0(v11[0]);
  }
}
