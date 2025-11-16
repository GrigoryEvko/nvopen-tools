// Function: sub_16B4110
// Address: 0x16b4110
//
__int64 __fastcall sub_16B4110(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  const char *v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  _WORD *v21; // rdx
  __int64 v22; // rdi
  __int64 result; // rax
  const char *v24; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v25; // [rsp+8h] [rbp-68h]
  _QWORD v26[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v27[4]; // [rsp+20h] [rbp-50h] BYREF
  int v28; // [rsp+40h] [rbp-30h]
  const char **v29; // [rsp+48h] [rbp-28h]

  sub_16B2F80(a1, a2, a5, a4);
  v24 = (const char *)v26;
  v25 = 0;
  LOBYTE(v26[0]) = 0;
  v27[0] = &unk_49EFBE0;
  v28 = 1;
  memset(&v27[1], 0, 24);
  v29 = &v24;
  sub_16E7A90(v27, a3);
  sub_16E7BC0(v27);
  v8 = sub_16E8C20(v27, a3, v7);
  v9 = *(_WORD **)(v8 + 24);
  v10 = v8;
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 1u )
  {
    v10 = sub_16E7EE0(v8, "= ", 2);
  }
  else
  {
    *v9 = 8253;
    *(_QWORD *)(v8 + 24) += 2LL;
  }
  v11 = 0;
  sub_16E7EE0(v10, v24, v25);
  if ( v25 < 8 )
    v11 = 8 - v25;
  v13 = sub_16E8C20(v10, 8 - v25, v12);
  v14 = sub_16E8750(v13, v11);
  sub_1263B40(v14, " (default: ");
  if ( *(_BYTE *)(a4 + 12) )
  {
    v16 = sub_16E8C20(v14, " (default: ", v15);
    v17 = (const char *)*(unsigned int *)(a4 + 8);
    v18 = v16;
    sub_16E7A90(v16, v17);
  }
  else
  {
    v17 = "*no default*";
    v18 = sub_16E8C20(v14, " (default: ", v15);
    sub_1263B40(v18, "*no default*");
  }
  v20 = sub_16E8C20(v18, v17, v19);
  v21 = *(_WORD **)(v20 + 24);
  v22 = v20;
  if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 1u )
  {
    result = sub_16E7EE0(v20, ")\n", 2);
  }
  else
  {
    result = 2601;
    *v21 = 2601;
    *(_QWORD *)(v22 + 24) += 2LL;
  }
  if ( v24 != (const char *)v26 )
    return j_j___libc_free_0(v24, v26[0] + 1LL);
  return result;
}
