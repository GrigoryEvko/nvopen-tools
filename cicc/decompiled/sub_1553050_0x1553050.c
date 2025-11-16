// Function: sub_1553050
// Address: 0x1553050
//
_BYTE *__fastcall sub_1553050(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r8
  char v8; // al
  unsigned __int8 v9; // di
  char v10; // al
  char v11; // al
  size_t v12; // rdx
  const char *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  _WORD *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdi
  _BYTE *result; // rax
  const char *v20[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+10h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_15E4B00(a2) )
    sub_1263B40(*a1, "; Materializable\n");
  sub_1550E20(*a1, a2, (__int64)(a1 + 5), a1[4], *(_QWORD *)(a2 + 40));
  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v5) <= 2 )
  {
    sub_16E7EE0(v4, " = ", 3);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 32;
    *(_WORD *)v5 = 15648;
    *(_QWORD *)(v4 + 24) += 3LL;
  }
  v6 = *a1;
  sub_1549EC0((__int64)v20, *(_BYTE *)(a2 + 32) & 0xF);
  sub_16E7EE0(v6, v20[0], v20[1]);
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0], v21 + 1);
  sub_154A4E0(a2, *a1);
  v7 = *a1;
  v8 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  if ( v8 != 1 )
  {
    if ( v8 == 2 )
    {
      sub_1263B40(*a1, "protected ");
      v7 = *a1;
    }
    v9 = *(_BYTE *)(a2 + 33);
    v10 = v9 & 3;
    if ( (v9 & 3) != 1 )
      goto LABEL_11;
LABEL_25:
    sub_1263B40(v7, "dllimport ");
    v7 = *a1;
    v9 = *(_BYTE *)(a2 + 33);
    goto LABEL_13;
  }
  sub_1263B40(*a1, "hidden ");
  v9 = *(_BYTE *)(a2 + 33);
  v7 = *a1;
  v10 = v9 & 3;
  if ( (v9 & 3) == 1 )
    goto LABEL_25;
LABEL_11:
  if ( v10 == 2 )
  {
    sub_1263B40(v7, "dllexport ");
    v7 = *a1;
    v9 = *(_BYTE *)(a2 + 33);
  }
LABEL_13:
  sub_154A050((v9 >> 2) & 7, v7);
  v11 = *(_BYTE *)(a2 + 32) >> 6;
  if ( v11 == 1 )
  {
    v12 = 18;
    v13 = "local_unnamed_addr";
  }
  else
  {
    v12 = 12;
    v13 = "unnamed_addr";
    if ( v11 != 2 )
      goto LABEL_16;
  }
  v14 = sub_1549FF0(*a1, v13, v12);
  sub_1549FC0(v14, 0x20u);
LABEL_16:
  if ( *(_BYTE *)(a2 + 16) == 1 )
    sub_1263B40(*a1, "alias ");
  else
    sub_1263B40(*a1, "ifunc ");
  sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)(a2 + 24), *a1);
  v15 = *a1;
  v16 = *(_WORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v16 <= 1u )
  {
    sub_16E7EE0(v15, ", ", 2);
  }
  else
  {
    *v16 = 8236;
    *(_QWORD *)(v15 + 24) += 2LL;
  }
  v17 = *(_QWORD *)(a2 - 24);
  if ( v17 )
  {
    sub_15520E0(a1, (__int64 *)v17, *(_BYTE *)(v17 + 16) != 5);
  }
  else
  {
    sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)a2, *a1);
    sub_1263B40(*a1, " <<NULL ALIASEE>>");
  }
  sub_1552170(a1, (const char *)a2);
  v18 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 16) )
    return (_BYTE *)sub_16E7DE0(v18, 10);
  *(_QWORD *)(v18 + 24) = result + 1;
  *result = 10;
  return result;
}
