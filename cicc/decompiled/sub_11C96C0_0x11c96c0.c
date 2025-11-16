// Function: sub_11C96C0
// Address: 0x11c96c0
//
__int64 __fastcall sub_11C96C0(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // r10
  __int64 *v11; // rsi
  __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // ecx
  int *v21; // rdx
  int v22; // r10d
  int v23; // edx
  int v24; // r12d

  v7 = a3;
  v9 = a2[((unsigned __int64)a3 >> 6) + 1] & (1LL << a3);
  if ( v9 )
  {
    v9 = 0;
    v12 = 0;
  }
  else
  {
    v10 = *a2;
    if ( (((int)*(unsigned __int8 *)(*a2 + ((unsigned int)v7 >> 2)) >> (2 * (v7 & 3))) & 3) != 0 )
    {
      if ( (((int)*(unsigned __int8 *)(*a2 + ((unsigned int)v7 >> 2)) >> (2 * (v7 & 3))) & 3) == 3 )
      {
        v11 = (__int64 *)(16 * v7 + 77034272);
        v12 = *v11;
        v9 = v11[1];
        goto LABEL_5;
      }
      v18 = *(unsigned int *)(v10 + 160);
      v19 = *(_QWORD *)(v10 + 144);
      if ( (_DWORD)v18 )
      {
        v20 = (v18 - 1) & (37 * a3);
        v21 = (int *)(v19 + 40LL * v20);
        v22 = *v21;
        if ( a3 == *v21 )
        {
LABEL_15:
          v12 = *((_QWORD *)v21 + 1);
          v9 = *((_QWORD *)v21 + 2);
          goto LABEL_5;
        }
        v23 = 1;
        while ( v22 != -1 )
        {
          v24 = v23 + 1;
          v20 = (v18 - 1) & (v23 + v20);
          v21 = (int *)(v19 + 40LL * v20);
          v22 = *v21;
          if ( a3 == *v21 )
            goto LABEL_15;
          v23 = v24;
        }
      }
      v21 = (int *)(v19 + 40 * v18);
      goto LABEL_15;
    }
    v12 = 0;
  }
LABEL_5:
  v13 = v12;
  v15 = sub_BA8C10(a1, v12, v9, a4, a5);
  v16 = v14;
  if ( a3 == 355 )
  {
    if ( *(_BYTE *)(*a2 + 170) || *(_BYTE *)(*a2 + 168) )
    {
      v13 = 2;
      if ( !(unsigned __int8)sub_B2D640(v14, 2, 54) )
      {
        v13 = 2;
        sub_B2D3C0(v16, 2, 54);
      }
    }
  }
  else
  {
    if ( a3 > 0x163 )
    {
      if ( a3 == 393 )
      {
LABEL_10:
        if ( *(_BYTE *)(*a2 + 170) || *(_BYTE *)(*a2 + 168) )
        {
          v13 = 0;
          if ( !(unsigned __int8)sub_B2D640(v14, 0, 54) )
          {
            v13 = 0;
            sub_B2D3C0(v16, 0, 54);
          }
        }
        goto LABEL_12;
      }
      if ( a3 <= 0x189 )
      {
        if ( a3 != 356 && a3 != 361 )
          goto LABEL_12;
      }
      else if ( a3 != 460 )
      {
        goto LABEL_12;
      }
      goto LABEL_17;
    }
    if ( a3 != 186 )
    {
      if ( a3 <= 0xB9 )
        goto LABEL_12;
      if ( a3 == 283 )
        goto LABEL_10;
      if ( a3 - 329 > 2 )
        goto LABEL_12;
LABEL_17:
      if ( *(_BYTE *)(*a2 + 170) || *(_BYTE *)(*a2 + 168) )
      {
        v13 = 1;
        if ( !(unsigned __int8)sub_B2D640(v14, 1, 54) )
        {
          v13 = 1;
          sub_B2D3C0(v16, 1, 54);
        }
      }
      goto LABEL_12;
    }
    if ( *(_BYTE *)(*a2 + 171) || *(_BYTE *)(*a2 + 169) )
    {
      v13 = 54;
      if ( !(unsigned __int8)sub_B2D630(v14, 54) )
      {
        v13 = 54;
        sub_B2D390(v16, 54);
      }
    }
  }
LABEL_12:
  sub_11C9540(v16, v13);
  return v15;
}
