// Function: sub_30CCC80
// Address: 0x30ccc80
//
__int64 __fastcall sub_30CCC80(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  char *v4; // r13
  size_t v5; // rax
  __int64 v7[2]; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-B0h] BYREF
  __int64 *v9; // [rsp+20h] [rbp-A0h]
  __int64 v10; // [rsp+30h] [rbp-90h] BYREF
  __int64 v11[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+70h] [rbp-50h]
  _QWORD v14[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( *(_DWORD *)a2 == 0x80000000 )
  {
    sub_B18290(a1, "(cost=always)", 0xDu);
  }
  else if ( *(_DWORD *)a2 == 0x7FFFFFFF )
  {
    sub_B18290(a1, "(cost=never)", 0xCu);
  }
  else
  {
    sub_B18290(a1, "(cost=", 6u);
    sub_B16530(v7, "Cost", 4, *(_DWORD *)a2);
    v2 = sub_2445430(a1, (__int64)v7);
    sub_B18290(v2, ", threshold=", 0xCu);
    sub_B16530(v11, "Threshold", 9, *(_DWORD *)(a2 + 4));
    v3 = sub_2445430(v2, (__int64)v11);
    sub_B18290(v3, ")", 1u);
    if ( v13 != v14 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( (_QWORD *)v11[0] != v12 )
      j_j___libc_free_0(v11[0]);
    if ( v9 != &v10 )
      j_j___libc_free_0((unsigned __int64)v9);
    if ( (__int64 *)v7[0] != &v8 )
      j_j___libc_free_0(v7[0]);
  }
  v4 = *(char **)(a2 + 16);
  if ( v4 )
  {
    sub_B18290(a1, ": ", 2u);
    v5 = strlen(v4);
    sub_B16430((__int64)v11, "Reason", 6u, v4, v5);
    sub_2445430(a1, (__int64)v11);
    if ( v13 != v14 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( (_QWORD *)v11[0] != v12 )
      j_j___libc_free_0(v11[0]);
  }
  return a1;
}
