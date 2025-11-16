// Function: sub_A52400
// Address: 0xa52400
//
__int64 __fastcall sub_A52400(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  int *v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rsi
  int *v13; // rax
  int *v14; // rax
  int v15; // r14d
  int *i; // r13
  _DWORD *v17; // rdx
  __int64 v18; // rdx
  const char *v19; // rsi
  __int64 v21; // rdx

  v6 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v6) <= 2 )
  {
    sub_CB6200(a1, &unk_3F23A08, 3);
    if ( *(_BYTE *)(a2 + 8) != 18 )
      goto LABEL_3;
  }
  else
  {
    *(_BYTE *)(v6 + 2) = 60;
    *(_WORD *)v6 = 8236;
    *(_QWORD *)(a1 + 32) += 3LL;
    if ( *(_BYTE *)(a2 + 8) != 18 )
      goto LABEL_3;
  }
  sub_904010(a1, "vscale x ");
LABEL_3:
  v7 = sub_CB59D0(a1, a4);
  v8 = *(_QWORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 7u )
  {
    sub_CB6200(v7, " x i32> ", 8);
  }
  else
  {
    *v8 = 0x203E323369207820LL;
    *(_QWORD *)(v7 + 32) += 8LL;
  }
  v9 = 4 * a4;
  v10 = &a3[a4];
  v11 = v9 >> 4;
  v12 = v9 >> 2;
  if ( v11 > 0 )
  {
    v13 = a3;
    while ( 1 )
    {
      if ( *v13 )
        goto LABEL_12;
      if ( v13[1] )
        break;
      if ( v13[2] )
      {
        v13 += 2;
        goto LABEL_12;
      }
      if ( v13[3] )
      {
        v13 += 3;
        goto LABEL_12;
      }
      v13 += 4;
      if ( v13 == &a3[4 * v11] )
      {
        v21 = v10 - v13;
        goto LABEL_39;
      }
    }
    ++v13;
LABEL_12:
    if ( v10 != v13 )
    {
      v14 = a3;
      goto LABEL_18;
    }
LABEL_43:
    v19 = "zeroinitializer";
    return sub_904010(a1, v19);
  }
  v21 = v12;
  v13 = a3;
LABEL_39:
  if ( v21 != 2 )
  {
    if ( v21 != 3 )
    {
      if ( v21 != 1 )
        goto LABEL_43;
      goto LABEL_42;
    }
    if ( *v13 )
      goto LABEL_52;
    ++v13;
  }
  if ( *v13 )
    goto LABEL_52;
  ++v13;
LABEL_42:
  if ( !*v13 )
    goto LABEL_43;
LABEL_52:
  if ( v10 == v13 )
    goto LABEL_43;
  v14 = a3;
  if ( v11 <= 0 )
  {
LABEL_32:
    if ( v12 != 2 )
    {
      if ( v12 != 3 )
      {
        if ( v12 != 1 )
          goto LABEL_35;
LABEL_59:
        if ( *v14 == -1 )
          goto LABEL_35;
        goto LABEL_19;
      }
      if ( *v14 != -1 )
      {
LABEL_19:
        if ( v10 == v14 )
          goto LABEL_35;
        goto LABEL_20;
      }
      ++v14;
    }
    if ( *v14 == -1 )
    {
      ++v14;
      goto LABEL_59;
    }
    goto LABEL_19;
  }
  while ( 1 )
  {
LABEL_18:
    if ( *v14 != -1 )
      goto LABEL_19;
    if ( v14[1] != -1 )
    {
      if ( v10 == v14 + 1 )
        goto LABEL_35;
      goto LABEL_20;
    }
    if ( v14[2] != -1 )
    {
      if ( v10 == v14 + 2 )
        goto LABEL_35;
      goto LABEL_20;
    }
    if ( v14[3] != -1 )
      break;
    v14 += 4;
    if ( !--v11 )
    {
      v12 = v10 - v14;
      goto LABEL_32;
    }
  }
  if ( v10 == v14 + 3 )
  {
LABEL_35:
    v19 = "poison";
    return sub_904010(a1, v19);
  }
LABEL_20:
  sub_904010(a1, "<");
  if ( a3 != v10 )
  {
    v15 = *a3;
    for ( i = a3 + 1; ; ++i )
    {
      v17 = *(_DWORD **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v17 <= 3u )
      {
        sub_CB6200(a1, "i32 ", 4);
      }
      else
      {
        *v17 = 540160873;
        *(_QWORD *)(a1 + 32) += 4LL;
      }
      if ( v15 != -1 )
        break;
      v18 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v18) <= 5 )
      {
        sub_CB6200(a1, "poison", 6);
LABEL_23:
        if ( v10 == i )
          goto LABEL_30;
        goto LABEL_24;
      }
      *(_DWORD *)v18 = 1936289648;
      *(_WORD *)(v18 + 4) = 28271;
      *(_QWORD *)(a1 + 32) += 6LL;
      if ( v10 == i )
        goto LABEL_30;
LABEL_24:
      v15 = *i;
      sub_904010(a1, ", ");
    }
    sub_CB59F0(a1, v15);
    goto LABEL_23;
  }
LABEL_30:
  v19 = ">";
  return sub_904010(a1, v19);
}
