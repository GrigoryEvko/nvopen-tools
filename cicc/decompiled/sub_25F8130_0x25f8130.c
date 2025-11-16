// Function: sub_25F8130
// Address: 0x25f8130
//
char *__fastcall sub_25F8130(_BYTE *src, _BYTE *a2, char *a3, char *a4, _QWORD *a5)
{
  _BYTE *v5; // r9
  char *v7; // r12
  __int64 v9; // rsi
  int v10; // edi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v14; // of
  signed __int64 v15; // rax
  signed __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  int v19; // r10d
  __int64 v20; // rax
  signed __int64 v21; // r13
  char *v22; // r8
  bool v24; // cc

  v5 = src;
  v7 = a3;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v9 = *(_QWORD *)v5;
      v10 = 1;
      v11 = *(_QWORD *)v7;
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
      {
        if ( v17 > 0 )
        {
          if ( v19 == v10 )
            goto LABEL_16;
          goto LABEL_15;
        }
        v20 = 0x7FFFFFFFFFFFFFFFLL;
      }
      if ( v19 == v10 )
      {
        if ( v20 > v16 )
          goto LABEL_5;
        goto LABEL_16;
      }
LABEL_15:
      if ( v10 < v19 )
      {
LABEL_5:
        *a5 = v11;
        v7 += 8;
        ++a5;
        if ( v5 == a2 )
          break;
        continue;
      }
LABEL_16:
      v5 += 8;
      *a5++ = v9;
      if ( v5 == a2 )
        break;
    }
    while ( v7 != a4 );
  }
  v21 = a2 - v5;
  if ( a2 != v5 )
    a5 = memmove(a5, v5, a2 - v5);
  v22 = (char *)a5 + v21;
  if ( a4 != v7 )
    v22 = (char *)memmove(v22, v7, a4 - v7);
  return &v22[a4 - v7];
}
