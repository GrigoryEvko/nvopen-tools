// Function: sub_A651F0
// Address: 0xa651f0
//
void (*__fastcall sub_A651F0(__int64 *a1, __int64 a2))()
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  void (*v6)(); // rcx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r14
  char v13; // al
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _DWORD *v17; // rdx
  __int64 v18; // rdi
  void (*result)(); // rax
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rbx
  __int64 v24; // rdi
  _WORD *v25; // rdx
  unsigned __int8 *v26; // rax
  size_t v27; // rdx
  unsigned __int8 *v28; // rax
  size_t v29; // rdx
  __int64 v30; // [rsp+10h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 72) )
  {
    if ( (unsigned __int8)sub_AA5B70(a2) )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
      {
        sub_904010(*a1, "\n");
        v28 = (unsigned __int8 *)sub_BD5D20(a2);
        sub_A54F00(*a1, v28, v29);
        sub_A51310(*a1, 0x3Au);
      }
      goto LABEL_5;
    }
    if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
      goto LABEL_28;
LABEL_42:
    sub_904010(*a1, "\n");
    v26 = (unsigned __int8 *)sub_BD5D20(a2);
    sub_A54F00(*a1, v26, v27);
    sub_A51310(*a1, 0x3Au);
    goto LABEL_30;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
    goto LABEL_42;
LABEL_28:
  sub_904010(*a1, "\n");
  v20 = sub_A5A650(a1[4], a2);
  if ( v20 == -1 )
  {
    sub_904010(*a1, "<badref>:");
  }
  else
  {
    v21 = sub_CB59F0(*a1, v20);
    sub_904010(v21, ":");
  }
LABEL_30:
  sub_C66A60(*a1, 50);
  sub_904010(*a1, ";");
  v22 = *(_QWORD *)(a2 + 16);
  if ( v22 )
  {
    while ( (unsigned __int8)(**(_BYTE **)(v22 + 24) - 30) > 0xAu )
    {
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        goto LABEL_44;
    }
    sub_904010(*a1, " preds = ");
    sub_A5B360(a1, *(_QWORD *)(*(_QWORD *)(v22 + 24) + 40LL), 0);
    v23 = *(_QWORD *)(v22 + 8);
    if ( v23 )
    {
      while ( (unsigned __int8)(**(_BYTE **)(v23 + 24) - 30) > 0xAu )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_5;
      }
LABEL_36:
      v24 = *a1;
      v25 = *(_WORD **)(*a1 + 32);
      if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v25 <= 1u )
      {
        sub_CB6200(v24, ", ", 2);
      }
      else
      {
        *v25 = 8236;
        *(_QWORD *)(v24 + 32) += 2LL;
      }
      sub_A5B360(a1, *(_QWORD *)(*(_QWORD *)(v23 + 24) + 40LL), 0);
      while ( 1 )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          break;
        if ( (unsigned __int8)(**(_BYTE **)(v23 + 24) - 30) <= 0xAu )
          goto LABEL_36;
      }
    }
  }
  else
  {
LABEL_44:
    sub_904010(*a1, " No predecessors!");
  }
LABEL_5:
  sub_904010(*a1, "\n");
  v3 = a1[33];
  v30 = a2 + 48;
  if ( v3 )
  {
    v4 = *(_QWORD *)v3;
    v5 = *a1;
    v6 = *(void (**)())(*(_QWORD *)v3 + 24LL);
    if ( v6 == nullsub_31 )
    {
      v7 = *(_QWORD *)(a2 + 56);
      if ( v30 != v7 )
        goto LABEL_8;
LABEL_25:
      result = *(void (**)())(v4 + 32);
      if ( result != nullsub_32 )
        return (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v3, a2, v5);
      return result;
    }
    result = (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))v6)(v3, a2, v5);
    v7 = *(_QWORD *)(a2 + 56);
    if ( v7 == v30 )
    {
LABEL_23:
      v3 = a1[33];
      if ( !v3 )
        return result;
      v4 = *(_QWORD *)v3;
      v5 = *a1;
      goto LABEL_25;
    }
    while ( 1 )
    {
LABEL_8:
      if ( !v7 )
LABEL_55:
        BUG();
      v8 = *(_QWORD *)(v7 + 40);
      if ( v8 )
      {
        v9 = sub_B14240(v8);
        v11 = v10;
        v12 = v9;
        if ( v9 != v10 )
          break;
      }
LABEL_20:
      sub_A635F0(a1, (unsigned __int8 *)(v7 - 24));
      v18 = *a1;
      result = *(void (**)())(*a1 + 32);
      if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
      {
        result = (void (*)())sub_CB5D20(v18, 10);
      }
      else
      {
        *(_QWORD *)(v18 + 32) = (char *)result + 1;
        *(_BYTE *)result = 10;
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( v7 == v30 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v16 = *a1;
        v17 = *(_DWORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v17 <= 3u )
          break;
        *v17 = 538976288;
        *(_QWORD *)(v16 + 32) += 4LL;
        v13 = *(_BYTE *)(v12 + 32);
        if ( v13 )
          goto LABEL_17;
LABEL_13:
        sub_A5C700(a1, v12);
        v14 = *a1;
        v15 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v15 < *(_QWORD *)(*a1 + 24) )
          goto LABEL_14;
LABEL_19:
        sub_CB5D20(v14, 10);
        v12 = *(_QWORD *)(v12 + 8);
        if ( v12 == v11 )
          goto LABEL_20;
      }
      sub_CB6200(v16, "    ", 4);
      v13 = *(_BYTE *)(v12 + 32);
      if ( !v13 )
        goto LABEL_13;
LABEL_17:
      if ( v13 != 1 )
        goto LABEL_55;
      sub_A5C590(a1, v12);
      v14 = *a1;
      v15 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(*a1 + 24) )
        goto LABEL_19;
LABEL_14:
      *(_QWORD *)(v14 + 32) = v15 + 1;
      *v15 = 10;
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v11 )
        goto LABEL_20;
    }
  }
  result = (void (*)())a2;
  v7 = *(_QWORD *)(a2 + 56);
  if ( v30 != v7 )
    goto LABEL_8;
  return result;
}
