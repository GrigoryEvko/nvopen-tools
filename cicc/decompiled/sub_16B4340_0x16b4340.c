// Function: sub_16B4340
// Address: 0x16b4340
//
__int64 __fastcall sub_16B4340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  void *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  const char *v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 result; // rax
  __int64 v24; // rax
  const char *v25; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v26; // [rsp+8h] [rbp-68h]
  _QWORD v27[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v28[4]; // [rsp+20h] [rbp-50h] BYREF
  int v29; // [rsp+40h] [rbp-30h]
  const char **v30; // [rsp+48h] [rbp-28h]

  sub_16B2F80(a1, a2, a5, a4);
  v25 = (const char *)v27;
  v26 = 0;
  LOBYTE(v27[0]) = 0;
  v28[0] = &unk_49EFBE0;
  v29 = 1;
  memset(&v28[1], 0, 24);
  v30 = &v25;
  sub_16E7AD0(v28);
  sub_16E7BC0(v28);
  v8 = sub_16E8C20(v28, a3, v7);
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
  sub_16E7EE0(v10, v25, v26);
  if ( v26 < 8 )
    v11 = 8 - v26;
  v13 = sub_16E8C20(v10, 8 - v26, v12);
  v14 = v11;
  v15 = sub_16E8750(v13, v11);
  v16 = *(void **)(v15 + 24);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 0xAu )
  {
    v14 = (unsigned __int64)" (default: ";
    sub_16E7EE0(v15, " (default: ", 11);
  }
  else
  {
    qmemcpy(v16, " (default: ", 11);
    *(_QWORD *)(v15 + 24) += 11LL;
  }
  if ( *(_BYTE *)(a4 + 16) )
  {
    v18 = sub_16E8C20(v17, v14, v16);
    v19 = *(const char **)(a4 + 8);
    v20 = v18;
    sub_16E7AD0(v18);
  }
  else
  {
    v24 = sub_16E8C20(v17, v14, v16);
    v19 = "*no default*";
    v20 = v24;
    sub_1263B40(v24, "*no default*");
  }
  v22 = sub_16E8C20(v20, v19, v21);
  result = sub_1263B40(v22, ")\n");
  if ( v25 != (const char *)v27 )
    return j_j___libc_free_0(v25, v27[0] + 1LL);
  return result;
}
