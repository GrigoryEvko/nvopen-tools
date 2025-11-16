// Function: sub_25F7FC0
// Address: 0x25f7fc0
//
char *__fastcall sub_25F7FC0(char *src, _BYTE *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v5; // r9
  __int64 v9; // rsi
  int v10; // edi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v14; // of
  signed __int64 v15; // rax
  signed __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  int v19; // r11d
  __int64 v20; // rax
  char *v21; // r8
  signed __int64 v22; // r12
  bool v24; // cc

  v5 = src;
  while ( a2 != v5 )
  {
    while ( 1 )
    {
      if ( a4 == a3 )
      {
        v21 = (char *)memmove(a5, v5, a2 - v5) + a2 - v5;
        v22 = 0;
        return &v21[v22];
      }
      v9 = *(_QWORD *)v5;
      v10 = 1;
      v11 = *(_QWORD *)a3;
      v12 = *(_QWORD *)(*(_QWORD *)v5 + 296LL);
      v13 = *(_QWORD *)(*(_QWORD *)v5 + 280LL);
      if ( *(_DWORD *)(*(_QWORD *)v5 + 304LL) != 1 )
        v10 = *(_DWORD *)(v9 + 288);
      v14 = __OFSUB__(v13, v12);
      v15 = v13 - v12;
      if ( v14 )
      {
        v24 = v12 <= 0;
        v16 = 0x8000000000000000LL;
        if ( v24 )
          v16 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v16 = v15;
      }
      v17 = *(_QWORD *)(v11 + 296);
      v18 = *(_QWORD *)(v11 + 280);
      v19 = 1;
      if ( *(_DWORD *)(v11 + 304) != 1 )
        v19 = *(_DWORD *)(v11 + 288);
      v14 = __OFSUB__(v18, v17);
      v20 = v18 - v17;
      if ( v14 )
        break;
LABEL_13:
      if ( v19 != v10 )
        goto LABEL_14;
      if ( v20 > v16 )
        goto LABEL_4;
LABEL_15:
      v5 += 8;
      *a5++ = v9;
      if ( a2 == v5 )
        goto LABEL_16;
    }
    if ( v17 <= 0 )
    {
      v20 = 0x7FFFFFFFFFFFFFFFLL;
      goto LABEL_13;
    }
    if ( v19 == v10 )
      goto LABEL_15;
LABEL_14:
    if ( v10 >= v19 )
      goto LABEL_15;
LABEL_4:
    *a5 = v11;
    a3 += 8;
    ++a5;
  }
LABEL_16:
  v21 = (char *)a5 + a2 - v5;
  v22 = a4 - a3;
  if ( a4 != a3 )
    v21 = (char *)memmove(v21, a3, a4 - a3);
  return &v21[v22];
}
