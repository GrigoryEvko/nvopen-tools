// Function: sub_D49BF0
// Address: 0xd49bf0
//
unsigned __int64 __fastcall sub_D49BF0(__int64 a1, __int64 a2, char a3, char a4, int a5)
{
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  _QWORD *v13; // rax
  unsigned int v14; // edx
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 result; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rax
  __int64 v22; // r14
  int v23; // r13d
  __int64 v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // r15
  __int64 v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rdx
  __int64 *v31; // r13
  __int64 *v32; // rbx
  __int64 v33; // rdi
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned int v39; // [rsp+24h] [rbp-3Ch]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v6 = (unsigned int)(2 * a5);
  v7 = a1;
  sub_CB69B0(a2, v6);
  if ( (unsigned __int8)sub_D497B0(a1, v6, v8, v9, v10, v11) )
    sub_904010(a2, "Parallel ");
  v12 = sub_904010(a2, "Loop at depth ");
  v13 = *(_QWORD **)v7;
  if ( *(_QWORD *)v7 )
  {
    v14 = 1;
    do
    {
      v13 = (_QWORD *)*v13;
      ++v14;
    }
    while ( v13 );
    v15 = v14;
  }
  else
  {
    v15 = 1;
  }
  v16 = sub_CB59D0(v12, v15);
  sub_904010(v16, " containing: ");
  result = *(_QWORD *)(v7 + 32);
  v36 = *(_QWORD *)result;
  if ( result == *(_QWORD *)(v7 + 40) )
    goto LABEL_43;
  v40 = a2;
  v18 = *(_QWORD *)result;
  v39 = 0;
  v37 = v7 + 56;
  if ( !a3 )
  {
LABEL_9:
    if ( v39 )
      sub_904010(v40, ",");
    sub_A5BF40((unsigned __int8 *)v18, v40, 0, 0);
    if ( v36 != v18 )
      goto LABEL_12;
    goto LABEL_34;
  }
  while ( 2 )
  {
    sub_904010(v40, "\n");
    if ( v36 == v18 )
LABEL_34:
      sub_904010(v40, "<header>");
LABEL_12:
    v19 = *(_QWORD *)(**(_QWORD **)(v7 + 32) + 16LL);
    if ( v19 )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v19 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
          break;
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_18;
      }
LABEL_16:
      if ( v18 == *(_QWORD *)(v20 + 40) )
      {
        sub_904010(v40, "<latch>");
      }
      else
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            break;
          v20 = *(_QWORD *)(v19 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
            goto LABEL_16;
        }
      }
    }
LABEL_18:
    v21 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v21 == v18 + 48 )
      goto LABEL_30;
    if ( !v21 )
      BUG();
    v22 = v21 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v21 - 24) - 30 > 0xA || (v23 = sub_B46E30(v22)) == 0 )
    {
LABEL_30:
      if ( a3 )
        goto LABEL_39;
      goto LABEL_31;
    }
    v24 = v7;
    v25 = 0;
    v26 = v24;
    while ( 1 )
    {
      v27 = sub_B46EC0(v22, v25);
      if ( !*(_BYTE *)(v26 + 84) )
        break;
      v28 = *(_QWORD **)(v26 + 64);
      v29 = &v28[*(unsigned int *)(v26 + 76)];
      if ( v28 == v29 )
        goto LABEL_38;
      while ( v27 != *v28 )
      {
        if ( v29 == ++v28 )
          goto LABEL_38;
      }
LABEL_28:
      if ( v23 == ++v25 )
      {
        v7 = v26;
        goto LABEL_30;
      }
    }
    if ( sub_C8CA60(v37, v27) )
      goto LABEL_28;
LABEL_38:
    v7 = v26;
    sub_904010(v40, "<exiting>");
    if ( a3 )
LABEL_39:
      sub_A68DD0(v18, v40, 0, 0, 0);
LABEL_31:
    v30 = *(_QWORD *)(v7 + 32);
    ++v39;
    result = (*(_QWORD *)(v7 + 40) - v30) >> 3;
    if ( v39 < result )
    {
      v18 = *(_QWORD *)(v30 + 8LL * v39);
      if ( !a3 )
        goto LABEL_9;
      continue;
    }
    break;
  }
  a2 = v40;
LABEL_43:
  if ( a4 )
  {
    sub_904010(a2, "\n");
    result = *(_QWORD *)(v7 + 8);
    v31 = *(__int64 **)(v7 + 16);
    if ( v31 != (__int64 *)result )
    {
      v32 = *(__int64 **)(v7 + 8);
      do
      {
        v33 = *v32++;
        result = sub_D49BF0(v33, a2, 0, 1, (unsigned int)(a5 + 2));
      }
      while ( v31 != v32 );
    }
  }
  return result;
}
