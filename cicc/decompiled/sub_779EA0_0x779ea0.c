// Function: sub_779EA0
// Address: 0x779ea0
//
__int64 __fastcall sub_779EA0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  char *v4; // r12
  char v6; // al
  __int64 result; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 m; // r8
  unsigned int n; // edx
  __int64 v11; // rax
  int v12; // r14d
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r8
  unsigned __int64 i; // r10
  unsigned int j; // edx
  __int64 v17; // rax
  char *v18; // rsi
  unsigned __int64 v19; // r8
  unsigned int k; // edx
  __int64 v21; // rax
  char *v22; // rsi
  int v23; // eax
  unsigned __int64 v24; // r8
  char ii; // al
  __int64 v26; // rsi
  unsigned int v27; // edx
  __int64 v28; // r15
  int v29; // r14d
  int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-58h]
  unsigned __int64 v33; // [rsp+18h] [rbp-48h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+18h] [rbp-48h]
  int v36[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v4 = a2;
  v6 = *(_BYTE *)(a3 + 140);
  if ( (unsigned __int8)(v6 - 9) <= 1u )
  {
    if ( (*(_BYTE *)(a3 + 176) & 8) == 0 )
      return 0;
    v12 = 0;
    v13 = **(_QWORD **)(a3 + 168);
    v14 = sub_76FF70(*(_QWORD *)(a3 + 160));
    if ( v14 )
    {
      do
      {
        for ( i = *(_QWORD *)(v14 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        for ( j = qword_4F08388 & (v14 >> 3); ; j = qword_4F08388 & (j + 1) )
        {
          v17 = qword_4F08380 + 16LL * j;
          if ( *(_QWORD *)v17 == v14 )
          {
            v18 = &v4[*(unsigned int *)(v17 + 8)];
            goto LABEL_28;
          }
          if ( !*(_QWORD *)v17 )
            break;
        }
        v18 = v4;
LABEL_28:
        v33 = v14;
        if ( (*(_BYTE *)(v14 + 144) & 0x20) != 0 )
        {
          sub_777E50(a1, (int)v18, i, a4);
          v19 = v33;
          v12 = 1;
        }
        else
        {
          v23 = sub_779EA0(a1, v18, i, a4);
          v19 = v33;
          if ( v23 )
            v12 = 1;
        }
        v14 = sub_76FF70(*(_QWORD *)(v19 + 112));
      }
      while ( v14 );
      if ( !v13 )
      {
LABEL_32:
        if ( !v12 )
          return 0;
LABEL_33:
        *(_BYTE *)(a4 - 9) &= ~1u;
        return 1;
      }
    }
    else if ( !v13 )
    {
      return 0;
    }
    do
    {
      if ( (*(_BYTE *)(v13 + 96) & 1) != 0 )
      {
        for ( k = qword_4F08388 & (v13 >> 3); ; k = qword_4F08388 & (k + 1) )
        {
          v21 = qword_4F08380 + 16LL * k;
          if ( *(_QWORD *)v21 == v13 )
          {
            v22 = &v4[*(unsigned int *)(v21 + 8)];
            goto LABEL_38;
          }
          if ( !*(_QWORD *)v21 )
            break;
        }
        v22 = v4;
LABEL_38:
        if ( (unsigned int)sub_779EA0(a1, v22, *(_QWORD *)(v13 + 40), a4) )
          v12 = 1;
      }
      v13 = *(_QWORD *)v13;
    }
    while ( v13 );
    goto LABEL_32;
  }
  if ( v6 == 11 )
  {
    if ( (*(_BYTE *)(a3 + 176) & 8) == 0 )
      return 0;
    if ( ((unsigned __int8)(1 << (((_BYTE)a2 - a4) & 7))
        & *(_BYTE *)(a4 + -(((unsigned int)((_DWORD)a2 - a4) >> 3) + 10))) == 0 )
      return 0;
    v8 = *(_QWORD *)a2;
    if ( !*(_QWORD *)a2 )
      return 0;
    for ( m = *(_QWORD *)(v8 + 120); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
      ;
    for ( n = qword_4F08388 & (v8 >> 3); ; n = qword_4F08388 & (n + 1) )
    {
      v11 = qword_4F08380 + 16LL * n;
      if ( v8 == *(_QWORD *)v11 )
        break;
      if ( !*(_QWORD *)v11 )
        goto LABEL_16;
    }
    v4 = &a2[*(unsigned int *)(v11 + 8)];
LABEL_16:
    if ( (*(_BYTE *)(v8 + 144) & 0x20) != 0 )
    {
      sub_777E50(a1, (int)v4, m, a4);
    }
    else if ( !(unsigned int)sub_779EA0(a1, v4, m, a4) )
    {
      return 0;
    }
    *(_BYTE *)(a4 - 9) &= ~1u;
    return 1;
  }
  if ( v6 != 8 )
    return 0;
  v24 = *(_QWORD *)(a3 + 160);
  for ( ii = *(_BYTE *)(v24 + 140); ii == 12; ii = *(_BYTE *)(v24 + 140) )
    v24 = *(_QWORD *)(v24 + 160);
  v26 = *(_QWORD *)(a3 + 176);
  v36[0] = 1;
  v27 = 16;
  if ( (unsigned __int8)(ii - 2) > 1u )
  {
    v35 = v24;
    v31 = sub_7764B0(a1, v24, v36);
    v24 = v35;
    v27 = v31;
  }
  result = 0;
  if ( v26 )
  {
    v28 = 0;
    v29 = 0;
    v32 = v27;
    do
    {
      v34 = v24;
      v30 = sub_779EA0(a1, v4, v24, a4);
      v24 = v34;
      if ( v30 )
        v29 = 1;
      v4 += v32;
      ++v28;
    }
    while ( v26 != v28 );
    if ( !v29 )
      return 0;
    goto LABEL_33;
  }
  return result;
}
