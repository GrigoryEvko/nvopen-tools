// Function: sub_A5C700
// Address: 0xa5c700
//
__int64 __fastcall sub_A5C700(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  char v8; // al
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _WORD *v12; // rdx
  __int64 v13; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = a1[1];
  v5 = a1[4];
  v17[0] = (__int64)off_4979428;
  v6 = (__int64)(a1 + 5);
  v7 = *a1;
  v17[1] = v6;
  v17[2] = v5;
  v17[3] = v4;
  sub_904010(v7, "#dbg_");
  v8 = *(_BYTE *)(a2 + 64);
  if ( v8 == 1 )
  {
    sub_904010(*a1, "value");
  }
  else
  {
    if ( v8 != 2 )
    {
      if ( v8 )
        BUG();
      sub_904010(*a1, "declare");
      v9 = *a1;
      v10 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) != v10 )
        goto LABEL_5;
      goto LABEL_16;
    }
    sub_904010(*a1, "assign");
  }
  v9 = *a1;
  v10 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) != v10 )
  {
LABEL_5:
    *v10 = 40;
    ++*(_QWORD *)(v9 + 32);
    goto LABEL_6;
  }
LABEL_16:
  sub_CB6200(v9, "(", 1);
LABEL_6:
  sub_A5C090(*a1, *(_QWORD *)(a2 + 40), v17);
  sub_904010(*a1, ", ");
  sub_A5C090(*a1, *(_QWORD *)(a2 + 72), v17);
  v11 = *a1;
  v12 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v12 <= 1u )
  {
    sub_CB6200(v11, ", ", 2);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 32) += 2LL;
  }
  sub_A5C090(*a1, *(_QWORD *)(a2 + 80), v17);
  sub_904010(*a1, ", ");
  if ( *(_BYTE *)(a2 + 64) == 2 )
  {
    sub_A5C090(*a1, *(_QWORD *)(a2 + 56), v17);
    sub_904010(*a1, ", ");
    if ( *(_BYTE *)(a2 + 64) == 2 )
      v15 = *(_QWORD *)(a2 + 48);
    else
      v15 = *(_QWORD *)(a2 + 40);
    sub_A5C090(*a1, v15, v17);
    sub_904010(*a1, ", ");
    sub_A5C090(*a1, *(_QWORD *)(a2 + 88), v17);
    sub_904010(*a1, ", ");
  }
  v13 = *(_QWORD *)(a2 + 24);
  v16 = v13;
  if ( v13 )
  {
    sub_B96E90(&v16, v13, 1);
    v13 = v16;
  }
  sub_A5C090(*a1, v13, v17);
  if ( v16 )
    sub_B91220(&v16);
  return sub_904010(*a1, ")");
}
