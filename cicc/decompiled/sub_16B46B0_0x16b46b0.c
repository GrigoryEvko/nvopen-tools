// Function: sub_16B46B0
// Address: 0x16b46b0
//
__int64 __fastcall sub_16B46B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, float a5)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rdx
  __int64 v12; // rax
  const char *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdi
  __int64 result; // rax
  const char *v22; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-68h]
  _QWORD v24[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v25[4]; // [rsp+30h] [rbp-50h] BYREF
  int v26; // [rsp+50h] [rbp-30h]
  const char **v27; // [rsp+58h] [rbp-28h]

  sub_16B2F80(a1, a2, a4, a4);
  v22 = (const char *)v24;
  LOBYTE(v24[0]) = 0;
  v23 = 0;
  v26 = 1;
  v25[0] = &unk_49EFBE0;
  memset(&v25[1], 0, 24);
  v27 = &v22;
  sub_16E7B70(v25, a5);
  sub_16E7BC0(v25);
  v7 = sub_16E8C20(v25, a2, v6);
  v8 = *(_WORD **)(v7 + 24);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_16E7EE0(v7, "= ", 2);
  }
  else
  {
    *v8 = 8253;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  v10 = 0;
  sub_16E7EE0(v9, v22, v23);
  if ( v23 < 8 )
    v10 = 8 - v23;
  v12 = sub_16E8C20(v9, 8 - v23, v11);
  v13 = " (default: ";
  v14 = sub_16E8750(v12, v10);
  sub_1263B40(v14, " (default: ");
  if ( *(_BYTE *)(a3 + 12) )
  {
    v16 = sub_16E8C20(v14, " (default: ", v15);
    sub_16E7B70(v16, *(float *)(a3 + 8));
  }
  else
  {
    v13 = "*no default*";
    v16 = sub_16E8C20(v14, " (default: ", v15);
    sub_1263B40(v16, "*no default*");
  }
  v18 = sub_16E8C20(v16, v13, v17);
  v19 = *(_WORD **)(v18 + 24);
  v20 = v18;
  if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 1u )
  {
    result = sub_16E7EE0(v18, ")\n", 2);
  }
  else
  {
    result = 2601;
    *v19 = 2601;
    *(_QWORD *)(v20 + 24) += 2LL;
  }
  if ( v22 != (const char *)v24 )
    return j_j___libc_free_0(v22, v24[0] + 1LL);
  return result;
}
