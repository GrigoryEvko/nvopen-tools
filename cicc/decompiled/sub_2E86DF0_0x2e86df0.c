// Function: sub_2E86DF0
// Address: 0x2e86df0
//
void __fastcall sub_2E86DF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v5; // rsi
  _BYTE *v6; // rax
  int v7; // r11d
  unsigned __int64 v8; // r10
  _BYTE *v9; // rcx
  int v10; // r11d
  unsigned __int64 v11; // rbx
  int v12; // r10d
  __int64 v13; // rbx
  __int64 v14; // r10
  int v15; // esi
  unsigned __int64 v16; // r11
  int v17; // r10d
  __int64 v18; // r10
  int v19; // r10d
  int v20; // r10d
  int v21; // r10d
  __int64 v22; // r10
  int v23; // r10d
  int v24; // esi
  __int64 v25; // rbx

  if ( a1 != a3 )
  {
    v3 = *(_QWORD *)(a1 + 48);
    v5 = *(_QWORD *)(a3 + 48);
    v6 = (_BYTE *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v7 = v3 & 7;
      if ( v7 == 1 )
      {
        v8 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        v9 = (_BYTE *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_37;
        goto LABEL_5;
      }
      if ( v7 == 3 )
      {
        if ( v6[4] )
        {
          v8 = *(_QWORD *)&v6[8 * *(int *)v6 + 16];
          v9 = (_BYTE *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
LABEL_5:
            v10 = v5 & 7;
            if ( v10 == 1 )
            {
              if ( (_BYTE *)v8 != v9 )
                goto LABEL_36;
              goto LABEL_7;
            }
            if ( v10 != 3 )
            {
              if ( v8 )
                goto LABEL_65;
              goto LABEL_7;
            }
            v16 = (unsigned __int64)v9;
            if ( !v9[4] )
            {
              if ( v8 )
                goto LABEL_50;
              goto LABEL_7;
            }
            goto LABEL_53;
          }
          if ( v8 )
            goto LABEL_37;
          if ( !v6[5] )
            goto LABEL_79;
          goto LABEL_35;
        }
        v9 = (_BYTE *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          if ( !v6[5] )
          {
LABEL_80:
            if ( !v6[6] )
            {
              v9 = 0;
              goto LABEL_82;
            }
LABEL_13:
            v13 = *(_QWORD *)&v6[8 * *(int *)v6 + 16 + 8 * (__int64)((unsigned __int8)v6[5] + (unsigned __int8)v6[4])];
            if ( v9 )
            {
              v10 = v5 & 7;
              if ( v10 != 3 )
              {
                if ( v13 )
                  goto LABEL_65;
                goto LABEL_16;
              }
LABEL_71:
              v22 = 0;
              v16 = (unsigned __int64)v9;
              if ( v9[6] )
                v22 = *(_QWORD *)&v9[8 * *(int *)v9
                                   + 16
                                   + 8 * (__int64)((unsigned __int8)v9[5] + (unsigned __int8)v9[4])];
              if ( v13 != v22 )
                goto LABEL_50;
LABEL_16:
              if ( v6 )
              {
                if ( (v3 & 7) == 3 )
                {
LABEL_18:
                  if ( v6[7] )
                    goto LABEL_19;
                  if ( (v5 & 7) != 3 )
                    goto LABEL_24;
LABEL_109:
                  v14 = 0;
                  goto LABEL_100;
                }
LABEL_99:
                v14 = 0;
                if ( (v5 & 7) != 3 )
                  goto LABEL_86;
                goto LABEL_100;
              }
              goto LABEL_108;
            }
            if ( v13 )
              goto LABEL_37;
LABEL_82:
            if ( !v6[7] )
            {
              if ( !v6[9] )
                goto LABEL_36;
              goto LABEL_84;
            }
LABEL_19:
            v14 = *(_QWORD *)&v6[8 * (unsigned __int8)v6[6]
                               + 16
                               + 8 * *(int *)v6
                               + 8 * (__int64)((unsigned __int8)v6[5] + (unsigned __int8)v6[4])];
            if ( v9 )
            {
              v10 = v5 & 7;
              if ( v10 != 3 )
              {
                if ( !v14 )
                  goto LABEL_22;
                goto LABEL_65;
              }
LABEL_100:
              v25 = 0;
              v16 = (unsigned __int64)v9;
              if ( v9[7] )
                v25 = *(_QWORD *)&v9[8 * (unsigned __int8)v9[6]
                                   + 16
                                   + 8 * *(int *)v9
                                   + 8 * (__int64)((unsigned __int8)v9[5] + (unsigned __int8)v9[4])];
              if ( v25 != v14 )
                goto LABEL_50;
LABEL_22:
              if ( !v6 || (v3 & 7) != 3 )
                goto LABEL_86;
LABEL_24:
              if ( !v6[9] )
                goto LABEL_86;
              if ( *(_QWORD *)&v6[8 * (unsigned __int8)v6[7]
                                + 16
                                + 8 * (unsigned __int8)v6[6]
                                + 8 * *(int *)v6
                                + 8 * (__int64)((unsigned __int8)v6[5] + (unsigned __int8)v6[4])] )
              {
                v15 = v5 & 7;
                if ( v15 == 3 )
                {
                  v16 = (unsigned __int64)v9;
                  if ( v9[9]
                    && *(_QWORD *)&v9[8 * (unsigned __int8)v9[7]
                                    + 16
                                    + 8 * (unsigned __int8)v9[6]
                                    + 8 * *(int *)v9
                                    + 8 * (__int64)((unsigned __int8)v9[5] + (unsigned __int8)v9[4])] )
                  {
                    *(_QWORD *)(a1 + 48) = *(_QWORD *)(a3 + 48);
                    return;
                  }
                  goto LABEL_50;
                }
                if ( v15 )
                {
                  v9 = 0;
                  goto LABEL_37;
                }
                goto LABEL_66;
              }
              if ( v9 )
              {
LABEL_86:
                v24 = v5 & 7;
                if ( v24 )
                {
                  if ( v24 == 3 )
                  {
                    v16 = (unsigned __int64)v9;
                    goto LABEL_50;
                  }
                  goto LABEL_36;
                }
LABEL_66:
                *(_QWORD *)(a3 + 48) = v9;
                sub_2E86A90(a1, a2, (_QWORD *)(a3 + 48), 1);
                return;
              }
LABEL_37:
              sub_2E86A90(a1, a2, 0, (__int64)v9);
              return;
            }
            if ( v14 || !v6[9] )
              goto LABEL_37;
LABEL_84:
            v9 = 0;
            goto LABEL_37;
          }
LABEL_35:
          if ( *(_QWORD *)&v6[8 * (unsigned __int8)v6[4] + 16 + 8 * (__int64)*(int *)v6] )
            goto LABEL_36;
LABEL_79:
          v9 = 0;
          goto LABEL_80;
        }
      }
      else
      {
        v9 = (_BYTE *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_37;
      }
      v17 = v5 & 7;
      if ( v17 == 1 )
        goto LABEL_36;
      if ( v17 != 3 )
        goto LABEL_7;
      v16 = (unsigned __int64)v9;
      if ( !v9[4] )
        goto LABEL_7;
    }
    else
    {
      v9 = (_BYTE *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_37;
      v19 = v5 & 7;
      if ( v19 == 1 )
        goto LABEL_36;
      if ( v19 != 3 )
        goto LABEL_60;
      v16 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v9[4] )
        goto LABEL_60;
    }
    v8 = 0;
LABEL_53:
    if ( *(_QWORD *)&v9[8 * *(int *)v9 + 16] != v8 )
      goto LABEL_50;
    if ( v6 )
    {
LABEL_7:
      v11 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v12 = v3 & 7;
      if ( v12 == 2 )
        goto LABEL_8;
      if ( v12 == 3 )
      {
        if ( v6[5] )
        {
          v11 = *(_QWORD *)&v6[8 * (unsigned __int8)v6[4] + 16 + 8 * (__int64)*(int *)v6];
LABEL_8:
          v10 = v5 & 7;
          if ( v10 == 2 )
          {
            if ( (_BYTE *)v11 != v9 )
              goto LABEL_36;
            goto LABEL_10;
          }
          if ( v10 != 3 )
          {
            if ( v11 )
              goto LABEL_65;
LABEL_10:
            if ( v6 )
              goto LABEL_11;
LABEL_97:
            if ( (v5 & 7) != 3 )
            {
LABEL_108:
              v10 = v5 & 7;
              if ( v10 == 3 )
                goto LABEL_109;
LABEL_65:
              if ( !v10 )
                goto LABEL_66;
LABEL_36:
              v9 = 0;
              goto LABEL_37;
            }
            v13 = 0;
            goto LABEL_71;
          }
          v18 = 0;
          v16 = (unsigned __int64)v9;
          if ( !v9[5] )
            goto LABEL_49;
          goto LABEL_48;
        }
        v23 = v5 & 7;
        if ( v23 == 2 )
          goto LABEL_36;
        if ( v23 != 3 )
          goto LABEL_10;
      }
      else
      {
        v21 = v5 & 7;
        if ( v21 == 2 )
          goto LABEL_36;
        if ( v21 != 3 )
        {
LABEL_11:
          if ( (v3 & 7) != 3 )
          {
            v13 = 0;
            if ( (v5 & 7) != 3 )
              goto LABEL_99;
            goto LABEL_71;
          }
          if ( !v6[6] )
          {
            v13 = 0;
            if ( (v5 & 7) != 3 )
              goto LABEL_18;
            goto LABEL_71;
          }
          goto LABEL_13;
        }
      }
LABEL_62:
      v11 = 0;
      v16 = (unsigned __int64)v9;
      if ( !v9[5] )
        goto LABEL_10;
LABEL_48:
      v18 = *(_QWORD *)&v9[8 * (unsigned __int8)v9[4] + 16 + 8 * (__int64)*(int *)v9];
LABEL_49:
      if ( v18 == v11 )
        goto LABEL_10;
LABEL_50:
      sub_2E86A90(a1, a2, (_QWORD *)(v16 + 16), *(int *)v9);
      return;
    }
LABEL_60:
    v20 = v5 & 7;
    if ( v20 == 2 )
      goto LABEL_36;
    if ( v20 != 3 )
      goto LABEL_97;
    goto LABEL_62;
  }
}
