// Function: sub_1DD6540
// Address: 0x1dd6540
//
void __fastcall sub_1DD6540(__int64 a1, __int16 a2, int a3)
{
  char *v5; // r8
  char *v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // r9
  char *v9; // rdx
  int v10; // ecx
  bool v11; // zf
  char *v12; // rdx
  char *v13; // rsi

  v5 = *(char **)(a1 + 160);
  v6 = *(char **)(a1 + 152);
  v7 = (v5 - v6) >> 5;
  v8 = (v5 - v6) >> 3;
  if ( v7 > 0 )
  {
    v9 = &v6[32 * v7];
    while ( a2 != *(_WORD *)v6 )
    {
      if ( a2 == *((_WORD *)v6 + 4) )
      {
        v6 += 8;
        goto LABEL_8;
      }
      if ( a2 == *((_WORD *)v6 + 8) )
      {
        v6 += 16;
        goto LABEL_8;
      }
      if ( a2 == *((_WORD *)v6 + 12) )
      {
        v6 += 24;
        goto LABEL_8;
      }
      v6 += 32;
      if ( v9 == v6 )
      {
        v8 = (v5 - v6) >> 3;
        goto LABEL_15;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  if ( v8 == 2 )
  {
LABEL_22:
    if ( a2 != *(_WORD *)v6 )
    {
      v6 += 8;
      goto LABEL_18;
    }
    goto LABEL_8;
  }
  if ( v8 != 3 )
  {
    if ( v8 != 1 )
      return;
LABEL_18:
    if ( a2 != *(_WORD *)v6 )
      return;
    goto LABEL_8;
  }
  if ( a2 != *(_WORD *)v6 )
  {
    v6 += 8;
    goto LABEL_22;
  }
LABEL_8:
  if ( v5 != v6 )
  {
    v10 = ~a3;
    v11 = (v10 & *((_DWORD *)v6 + 1)) == 0;
    *((_DWORD *)v6 + 1) &= v10;
    if ( v11 )
    {
      v12 = *(char **)(a1 + 160);
      v13 = v6 + 8;
      if ( v12 != v6 + 8 )
      {
        memmove(v6, v13, v12 - v13);
        v13 = *(char **)(a1 + 160);
      }
      *(_QWORD *)(a1 + 160) = v13 - 8;
    }
  }
}
