// Function: sub_1EB3C10
// Address: 0x1eb3c10
//
__int64 __fastcall sub_1EB3C10(__int64 a1)
{
  char *v1; // rax
  char *v2; // rcx
  __int64 v3; // r10
  char *v5; // rax
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 i; // rsi
  __int64 v13; // rax
  char *v14; // rdx
  char *v15; // r11
  __int64 v16; // r14
  __int64 v17; // rsi
  char *v18; // r8
  char *v19; // rax

  v1 = *(char **)(a1 + 704);
  v2 = *(char **)(a1 + 696);
  if ( v2 == v1 )
    return 0;
  v3 = *(_QWORD *)v2;
  if ( v1 - v2 <= 8 )
  {
    *(_QWORD *)(a1 + 704) = v1 - 8;
    return v3;
  }
  v5 = v1 - 8;
  v6 = *(_QWORD *)v5;
  *(_QWORD *)v5 = v3;
  v7 = v5 - v2;
  v8 = v7 >> 3;
  v9 = (v7 >> 3) - 1;
  v10 = (v7 >> 3) & 1;
  v11 = v9 / 2;
  if ( v7 <= 16 )
  {
    v19 = v2;
    if ( !v10 && (unsigned __int64)v9 <= 2 )
    {
      v14 = v2;
      v13 = 0;
      goto LABEL_17;
    }
  }
  else
  {
    for ( i = 0; ; i = v13 )
    {
      v13 = 2 * (i + 1);
      v14 = &v2[16 * i + 16];
      v15 = &v2[8 * v13 - 8];
      v16 = *(_QWORD *)v14;
      if ( *(float *)(*(_QWORD *)v15 + 116LL) > *(float *)(*(_QWORD *)v14 + 116LL) )
      {
        v16 = *(_QWORD *)v15;
        v14 = &v2[8 * --v13];
      }
      *(_QWORD *)&v2[8 * i] = v16;
      if ( v13 >= v11 )
        break;
    }
    if ( v10 )
      goto LABEL_11;
    v17 = (v13 - 1) >> 1;
    if ( v13 == (v8 - 2) / 2 )
    {
LABEL_17:
      v13 = 2 * v13 + 1;
      *(_QWORD *)v14 = *(_QWORD *)&v2[8 * v13];
LABEL_11:
      v17 = (v13 - 1) >> 1;
    }
    while ( 1 )
    {
      v18 = &v2[8 * v17];
      v19 = &v2[8 * v13];
      if ( *(float *)(v6 + 116) <= *(float *)(*(_QWORD *)v18 + 116LL) )
        break;
      *(_QWORD *)v19 = *(_QWORD *)v18;
      v13 = v17;
      if ( !v17 )
      {
        v19 = v2;
        break;
      }
      v17 = (v17 - 1) / 2;
    }
  }
  *(_QWORD *)v19 = v6;
  *(_QWORD *)(a1 + 704) -= 8LL;
  return v3;
}
