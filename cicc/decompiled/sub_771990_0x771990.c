// Function: sub_771990
// Address: 0x771990
//
int __fastcall sub_771990(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // esi
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rbx
  int result; // eax
  _QWORD *j; // rbx
  _QWORD *v12; // rdi
  _QWORD *v13; // rdx
  _QWORD *i; // rax
  char *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  --qword_4F082A8;
  if ( v2 )
  {
    if ( qword_4F082A0 )
    {
      v3 = v2;
      do
      {
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 8);
      }
      while ( v3 );
      *(_QWORD *)(v4 + 8) = qword_4F082A0;
    }
    qword_4F082A0 = v2;
    *(_QWORD *)(a1 + 24) = 0;
  }
  sub_770440(a1);
  v5 = *(_DWORD *)(a1 + 64);
  *(_QWORD *)a1 = 0;
  if ( !v5 )
  {
    v7 = 0;
    goto LABEL_12;
  }
  v6 = v5;
  LODWORD(v7) = 0;
  do
  {
    v8 = (unsigned int)(v6 - 1);
    v7 = (unsigned int)(v7 + 1);
    v6 &= v8;
  }
  while ( v6 );
  if ( (int)v7 <= 10 )
  {
    v7 = (int)v7;
LABEL_12:
    **(_QWORD **)(a1 + 56) = qword_4F082C0[v7];
    qword_4F082C0[v7] = *(_QWORD *)(a1 + 56);
    goto LABEL_13;
  }
  sub_822B90(*(_QWORD *)(a1 + 56), (unsigned int)(4 * v5 + 4), v8, v7);
LABEL_13:
  v9 = *(_QWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 56) = 0;
  for ( v19[0] = v9; v9; v19[0] = v9 )
  {
    v9 = *(_QWORD *)(v9 + 120);
    sub_724E30((__int64)v19);
  }
  if ( (*(_BYTE *)(a1 + 132) & 8) != 0 )
  {
    v16 = *(_QWORD *)(a1 + 152);
    if ( v16 )
    {
      if ( qword_4F082A0 )
      {
        v17 = *(_QWORD *)(a1 + 152);
        do
        {
          v18 = v17;
          v17 = *(_QWORD *)(v17 + 8);
        }
        while ( v17 );
        *(_QWORD *)(v18 + 8) = qword_4F082A0;
      }
      qword_4F082A0 = v16;
      *(_QWORD *)(a1 + 152) = 0;
    }
  }
  result = qword_4F08090;
  if ( qword_4F08080 != qword_4F08090 && !qword_4F082A8 )
  {
    v13 = (_QWORD *)qword_4F08098;
    for ( i = *(_QWORD **)(qword_4F08098 + 8); i; i = (_QWORD *)i[1] )
    {
      *v13 = i;
      v13 = i;
    }
    *v13 = 0;
    qword_4F08088 = qword_4F08098;
    result = qword_4F08090;
    qword_4F08080 = qword_4F08090;
  }
  for ( j = *(_QWORD **)(a1 + 184); j; result = j___libc_free(v12, *((unsigned int *)v12 + 9)) )
  {
    v12 = j;
    j = (_QWORD *)*j;
  }
  if ( (*(_BYTE *)(a1 + 133) & 0x20) != 0 )
  {
    v15 = sub_67C860(2998);
    return fprintf(qword_4F07510, "\n%s\n", v15);
  }
  return result;
}
