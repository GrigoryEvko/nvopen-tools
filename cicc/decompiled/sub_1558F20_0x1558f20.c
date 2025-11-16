// Function: sub_1558F20
// Address: 0x1558f20
//
void (*__fastcall sub_1558F20(__int64 *a1, __int64 a2))()
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  void (*result)(); // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  void (*v16)(); // rcx
  __int64 i; // rbx
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdi
  _WORD *v25; // rdx
  const char *v26; // rax
  size_t v27; // rdx

  v4 = *a1;
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    sub_1263B40(v4, "\n");
    v26 = (const char *)sub_1649960(a2);
    sub_154B650(*a1, v26, v27);
    sub_1549FC0(*a1, 0x3Au);
    v9 = *(_QWORD *)(a2 + 56);
    v4 = *a1;
    if ( v9 )
      goto LABEL_7;
LABEL_38:
    sub_16BE270(v4, 50);
    sub_1263B40(*a1, "; Error: Block without parent!");
    goto LABEL_9;
  }
  if ( *(_QWORD *)(a2 + 8) )
  {
    sub_1263B40(v4, "\n; <label>:");
    v7 = sub_154F3B0(a1[4], a2, v5, v6);
    if ( v7 == -1 )
    {
      sub_1263B40(*a1, "<badref>");
    }
    else
    {
      v8 = sub_16E7AB0(*a1, v7);
      sub_1263B40(v8, ":");
    }
    v4 = *a1;
  }
  v9 = *(_QWORD *)(a2 + 56);
  if ( !v9 )
    goto LABEL_38;
LABEL_7:
  v10 = *(_QWORD *)(v9 + 80);
  if ( !v10 || a2 != v10 - 24 )
  {
    sub_16BE270(v4, 50);
    sub_1263B40(*a1, ";");
    v20 = *(_QWORD *)(a2 + 8);
    if ( v20 )
    {
      while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v20) + 16) - 25) > 9u )
      {
        v20 = *(_QWORD *)(v20 + 8);
        if ( !v20 )
          goto LABEL_34;
      }
      sub_1263B40(*a1, " preds = ");
      v21 = sub_1648700(v20);
      sub_15520E0(a1, *(__int64 **)(v21 + 40), 0);
      v22 = *(_QWORD *)(v20 + 8);
      if ( v22 )
      {
        while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v22) + 16) - 25) > 9u )
        {
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            goto LABEL_9;
        }
LABEL_31:
        v24 = *a1;
        v25 = *(_WORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v25 > 1u )
        {
          *v25 = 8236;
          *(_QWORD *)(v24 + 24) += 2LL;
        }
        else
        {
          sub_16E7EE0(v24, ", ", 2);
        }
        v23 = sub_1648700(v22);
        sub_15520E0(a1, *(__int64 **)(v23 + 40), 0);
        while ( 1 )
        {
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            break;
          if ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v22) + 16) - 25) <= 9u )
            goto LABEL_31;
        }
      }
    }
    else
    {
LABEL_34:
      sub_1263B40(*a1, " No predecessors!");
    }
  }
LABEL_9:
  v11 = a2 + 40;
  result = (void (*)())sub_1263B40(*a1, "\n");
  v13 = a1[29];
  if ( v13 )
  {
    v14 = *(_QWORD *)v13;
    v15 = *a1;
    v16 = *(void (**)())(*(_QWORD *)v13 + 24LL);
    if ( v16 == nullsub_536 )
    {
      i = *(_QWORD *)(a2 + 48);
      if ( i != v11 )
        goto LABEL_14;
      goto LABEL_20;
    }
    result = (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))v16)(v13, a2, v15);
    for ( i = *(_QWORD *)(a2 + 48); i != v11; i = *(_QWORD *)(i + 8) )
    {
      while ( 1 )
      {
LABEL_14:
        v18 = i - 24;
        if ( !i )
          v18 = 0;
        sub_15572A0(a1, v18);
        v19 = *a1;
        result = *(void (**)())(*a1 + 24);
        if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 16) )
          break;
        *(_QWORD *)(v19 + 24) = (char *)result + 1;
        *(_BYTE *)result = 10;
        i = *(_QWORD *)(i + 8);
        if ( i == v11 )
          goto LABEL_18;
      }
      result = (void (*)())sub_16E7DE0(v19, 10);
    }
LABEL_18:
    v13 = a1[29];
    if ( v13 )
    {
      v14 = *(_QWORD *)v13;
      v15 = *a1;
LABEL_20:
      result = *(void (**)())(v14 + 32);
      if ( result != nullsub_525 )
        return (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v13, a2, v15);
    }
  }
  else
  {
    i = *(_QWORD *)(a2 + 48);
    if ( i != v11 )
      goto LABEL_14;
  }
  return result;
}
