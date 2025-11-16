// Function: sub_A62950
// Address: 0xa62950
//
_BYTE *__fastcall sub_A62950(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rdi
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rdi
  _WORD *v14; // rdx
  _BYTE *v15; // rsi
  __int64 v16; // rdi
  _BYTE *result; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD v21[4]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v22[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+30h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_B2F600(a2) )
    sub_904010(*a1, "; Materializable\n");
  v4 = a1[4];
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *a1;
  v21[1] = a1 + 5;
  v21[2] = v4;
  v21[3] = v5;
  v21[0] = off_4979428;
  sub_A5A730(v6, a2, (__int64)v21);
  v7 = *a1;
  v8 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v8) <= 2 )
  {
    sub_CB6200(v7, " = ", 3);
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 32;
    *(_WORD *)v8 = 15648;
    *(_QWORD *)(v7 + 32) += 3LL;
  }
  v9 = *a1;
  sub_A51210((__int64)v22, *(_BYTE *)(a2 + 32) & 0xF);
  sub_CB6200(v9, v22[0], v22[1]);
  if ( (__int64 *)v22[0] != &v23 )
    j_j___libc_free_0(v22[0], v23 + 1);
  sub_A518A0(a2, *a1);
  v10 = *a1;
  v11 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  if ( v11 == 1 )
  {
    sub_904010(v10, "hidden ");
    v10 = *a1;
  }
  else if ( v11 == 2 )
  {
    sub_904010(v10, "protected ");
    v10 = *a1;
  }
  v12 = *(_QWORD *)(v10 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v12) <= 5 )
  {
    sub_CB6200(v10, "ifunc ", 6);
  }
  else
  {
    *(_DWORD *)v12 = 1853187689;
    *(_WORD *)(v12 + 4) = 8291;
    *(_QWORD *)(v10 + 32) += 6LL;
  }
  sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 24), *a1);
  v13 = *a1;
  v14 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v14 > 1u )
  {
    *v14 = 8236;
    *(_QWORD *)(v13 + 32) += 2LL;
    v15 = *(_BYTE **)(a2 - 32);
    if ( v15 )
      goto LABEL_14;
LABEL_21:
    sub_A57EC0((__int64)(a1 + 5), *(_QWORD *)(a2 + 8), *a1);
    sub_904010(*a1, " <<NULL RESOLVER>>");
    if ( *(char *)(a2 + 33) >= 0 )
      goto LABEL_15;
    goto LABEL_22;
  }
  sub_CB6200(v13, ", ", 2);
  v15 = *(_BYTE **)(a2 - 32);
  if ( !v15 )
    goto LABEL_21;
LABEL_14:
  sub_A5B360(a1, (__int64)v15, *v15 != 5);
  if ( *(char *)(a2 + 33) >= 0 )
    goto LABEL_15;
LABEL_22:
  sub_904010(*a1, ", partition \"");
  v18 = *a1;
  v19 = sub_B30A70(a2);
  sub_C92400(v19, v20, v18);
  sub_A51310(*a1, 0x22u);
LABEL_15:
  sub_A61E50(a1, a2);
  v16 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
    return (_BYTE *)sub_CB5D20(v16, 10);
  *(_QWORD *)(v16 + 32) = result + 1;
  *result = 10;
  return result;
}
