// Function: sub_1BBAC30
// Address: 0x1bbac30
//
char *__fastcall sub_1BBAC30(char *src, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *i; // r14
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rbx
  char *v15; // r12
  signed __int64 v16; // rbx
  __int64 v18; // [rsp+8h] [rbp-48h]
  char *v19; // [rsp+10h] [rbp-40h]
  char *v20; // [rsp+18h] [rbp-38h]

  for ( i = src; i != a2; ++a5 )
  {
    while ( 1 )
    {
      if ( a3 == a4 )
        goto LABEL_16;
      v9 = *(_QWORD *)a3;
      v10 = *(_QWORD *)i;
      if ( *(_QWORD *)i != 0 && *(_QWORD *)a3 != 0 && v10 != v9 )
      {
        if ( v9 == *(_QWORD *)(v10 + 8) )
          goto LABEL_25;
        if ( v10 != *(_QWORD *)(v9 + 8) && *(_DWORD *)(v9 + 16) < *(_DWORD *)(v10 + 16) )
          break;
      }
LABEL_23:
      i += 8;
      *a5++ = v10;
      if ( i == a2 )
        goto LABEL_16;
    }
    v11 = *(_QWORD *)(a6 + 1352);
    if ( *(_BYTE *)(v11 + 72) )
    {
      if ( *(_DWORD *)(v10 + 48) >= *(_DWORD *)(v9 + 48) && *(_DWORD *)(v10 + 52) <= *(_DWORD *)(v9 + 52) )
      {
LABEL_25:
        v14 = *(_QWORD *)a3;
        goto LABEL_15;
      }
      goto LABEL_23;
    }
    v12 = *(_DWORD *)(v11 + 76) + 1;
    *(_DWORD *)(v11 + 76) = v12;
    if ( v12 <= 0x20 )
    {
      do
      {
        v13 = v10;
        v10 = *(_QWORD *)(v10 + 8);
      }
      while ( v10 && *(_DWORD *)(v9 + 16) <= *(_DWORD *)(v10 + 16) );
      if ( v9 == v13 )
        goto LABEL_14;
LABEL_28:
      v10 = *(_QWORD *)i;
      goto LABEL_23;
    }
    v18 = a6;
    v19 = a4;
    sub_15CC640(v11);
    a4 = v19;
    a6 = v18;
    if ( *(_DWORD *)(v10 + 48) < *(_DWORD *)(v9 + 48) || *(_DWORD *)(v10 + 52) > *(_DWORD *)(v9 + 52) )
      goto LABEL_28;
LABEL_14:
    v14 = *(_QWORD *)a3;
LABEL_15:
    *a5 = v14;
    a3 += 8;
  }
LABEL_16:
  if ( a2 != i )
  {
    v20 = a4;
    memmove(a5, i, a2 - i);
    a4 = v20;
  }
  v15 = (char *)a5 + a2 - i;
  v16 = a4 - a3;
  if ( a4 != a3 )
    memmove(v15, a3, a4 - a3);
  return &v15[v16];
}
