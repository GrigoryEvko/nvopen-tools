// Function: sub_A62650
// Address: 0xa62650
//
_BYTE *__fastcall sub_A62650(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v8; // r8
  char v9; // al
  unsigned __int8 v10; // di
  char v11; // al
  char v12; // al
  size_t v13; // rdx
  char *v14; // rsi
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 v17; // rdi
  _BYTE *result; // rax
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD v22[4]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v23[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v24; // [rsp+30h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_B2F600(a2) )
    sub_904010(*a1, "; Materializable\n");
  v4 = a1[4];
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *a1;
  v22[1] = a1 + 5;
  v22[2] = v4;
  v22[0] = off_4979428;
  v22[3] = v5;
  sub_A5A730(v6, a2, (__int64)v22);
  sub_904010(*a1, " = ");
  v7 = *a1;
  sub_A51210((__int64)v23, *(_BYTE *)(a2 + 32) & 0xF);
  sub_CB6200(v7, v23[0], v23[1]);
  if ( (__int64 *)v23[0] != &v24 )
    j_j___libc_free_0(v23[0], v24 + 1);
  sub_A518A0(a2, *a1);
  v8 = *a1;
  v9 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  if ( v9 != 1 )
  {
    if ( v9 == 2 )
    {
      sub_904010(*a1, "protected ");
      v8 = *a1;
    }
    v10 = *(_BYTE *)(a2 + 33);
    v11 = v10 & 3;
    if ( (v10 & 3) != 1 )
      goto LABEL_9;
LABEL_21:
    sub_904010(v8, "dllimport ");
    v8 = *a1;
    v10 = *(_BYTE *)(a2 + 33);
    goto LABEL_11;
  }
  sub_904010(*a1, "hidden ");
  v10 = *(_BYTE *)(a2 + 33);
  v8 = *a1;
  v11 = v10 & 3;
  if ( (v10 & 3) == 1 )
    goto LABEL_21;
LABEL_9:
  if ( v11 == 2 )
  {
    sub_904010(v8, "dllexport ");
    v8 = *a1;
    v10 = *(_BYTE *)(a2 + 33);
  }
LABEL_11:
  sub_A513A0((v10 >> 2) & 7, v8);
  v12 = *(_BYTE *)(a2 + 32) >> 6;
  if ( v12 == 1 )
  {
    v13 = 18;
    v14 = "local_unnamed_addr";
    goto LABEL_13;
  }
  v13 = 12;
  v14 = "unnamed_addr";
  if ( v12 == 2 )
  {
LABEL_13:
    v15 = sub_A51340(*a1, v14, v13);
    sub_A51310(v15, 0x20u);
    goto LABEL_14;
  }
  if ( v12 )
    BUG();
LABEL_14:
  sub_904010(*a1, "alias ");
  sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 24), *a1);
  sub_904010(*a1, ", ");
  v16 = *(_BYTE **)(a2 - 32);
  if ( v16 )
  {
    sub_A5B360(a1, (__int64)v16, *v16 != 5);
    if ( *(char *)(a2 + 33) >= 0 )
      goto LABEL_16;
  }
  else
  {
    sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 8), *a1);
    sub_904010(*a1, " <<NULL ALIASEE>>");
    if ( *(char *)(a2 + 33) >= 0 )
      goto LABEL_16;
  }
  sub_904010(*a1, ", partition \"");
  v19 = *a1;
  v20 = sub_B30A70(a2);
  sub_C92400(v20, v21, v19);
  sub_A51310(*a1, 0x22u);
LABEL_16:
  sub_A61E50(a1, a2);
  v17 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
    return (_BYTE *)sub_CB5D20(v17, 10);
  *(_QWORD *)(v17 + 32) = result + 1;
  *result = 10;
  return result;
}
