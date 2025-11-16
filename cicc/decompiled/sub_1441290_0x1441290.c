// Function: sub_1441290
// Address: 0x1441290
//
__int64 __fastcall sub_1441290(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdi
  _WORD *v11; // rdx
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // rdi
  __int64 result; // rax
  __int64 *v21; // r14
  __int64 *v22; // rbx
  unsigned int i; // r15d
  __int64 v24; // rdi
  __int64 v25; // rax
  void *v26; // rdx

  v5 = sub_16E8750(a2, 2 * a3);
  v6 = *(_BYTE **)(v5 + 24);
  if ( *(_BYTE **)(v5 + 16) == v6 )
  {
    v5 = sub_16E7EE0(v5, "[", 1);
  }
  else
  {
    *v6 = 91;
    ++*(_QWORD *)(v5 + 24);
  }
  v7 = sub_16E7A90(v5, a3);
  v8 = *(_WORD **)(v7 + 24);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
  {
    v25 = sub_16E7EE0(v7, "] ", 2);
    v10 = *a1;
    v9 = v25;
    if ( *a1 )
      goto LABEL_5;
LABEL_23:
    v26 = *(void **)(v9 + 24);
    if ( *(_QWORD *)(v9 + 16) - (_QWORD)v26 <= 0xDu )
    {
      sub_16E7EE0(v9, " <<exit node>>", 14);
      v11 = *(_WORD **)(v9 + 24);
    }
    else
    {
      qmemcpy(v26, " <<exit node>>", 14);
      v11 = (_WORD *)(*(_QWORD *)(v9 + 24) + 14LL);
      *(_QWORD *)(v9 + 24) = v11;
    }
    goto LABEL_6;
  }
  *v8 = 8285;
  *(_QWORD *)(v7 + 24) += 2LL;
  v10 = *a1;
  if ( !*a1 )
    goto LABEL_23;
LABEL_5:
  sub_15537D0(v10, v9, 0);
  v11 = *(_WORD **)(v9 + 24);
LABEL_6:
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v11 <= 1u )
  {
    v9 = sub_16E7EE0(v9, " {", 2);
  }
  else
  {
    *v11 = 31520;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  v12 = sub_16E7A90(v9, *((unsigned int *)a1 + 12));
  v13 = *(_BYTE **)(v12 + 24);
  if ( *(_BYTE **)(v12 + 16) == v13 )
  {
    v12 = sub_16E7EE0(v12, ",", 1);
  }
  else
  {
    *v13 = 44;
    ++*(_QWORD *)(v12 + 24);
  }
  v14 = sub_16E7A90(v12, *((unsigned int *)a1 + 13));
  v15 = *(_QWORD *)(v14 + 24);
  v16 = v14;
  if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v15) <= 2 )
  {
    v16 = sub_16E7EE0(v14, "} [", 3);
  }
  else
  {
    *(_BYTE *)(v15 + 2) = 91;
    *(_WORD *)v15 = 8317;
    *(_QWORD *)(v14 + 24) += 3LL;
  }
  v17 = sub_16E7A90(v16, *((unsigned int *)a1 + 4));
  v18 = *(_WORD **)(v17 + 24);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 1u )
  {
    result = sub_16E7EE0(v17, "]\n", 2);
  }
  else
  {
    result = 2653;
    *v18 = 2653;
    *(_QWORD *)(v19 + 24) += 2LL;
  }
  v21 = (__int64 *)a1[4];
  v22 = (__int64 *)a1[3];
  for ( i = a3 + 1; v21 != v22; result = sub_1441290(v24, a2, i) )
    v24 = *v22++;
  return result;
}
