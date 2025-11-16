// Function: sub_85BD30
// Address: 0x85bd30
//
__int64 *__fastcall sub_85BD30(__int64 a1)
{
  __int64 *result; // rax
  char v2; // al
  char v3; // al
  char v4; // al
  char v5; // al
  __int64 *v6; // rbx
  char v7; // al
  __int64 *v8; // r15
  char v9; // al
  __int64 *v10; // r14
  char v11; // al
  __int64 *v12; // r13
  char v13; // al
  __int64 *v14; // r12
  char v15; // cl
  __int64 *v16; // [rsp+10h] [rbp-50h]
  __int64 *v17; // [rsp+18h] [rbp-48h]
  __int64 *v18; // [rsp+20h] [rbp-40h]
  __int64 *v19; // [rsp+28h] [rbp-38h]

  result = *(__int64 **)(a1 + 160);
  v16 = result;
  if ( result )
  {
    while ( 1 )
    {
      v2 = *((_BYTE *)v16 + 28);
      if ( v2 != 2 && v2 != 15 )
        goto LABEL_4;
      v16[2] = a1;
      v17 = (__int64 *)v16[20];
      if ( !v17 )
        goto LABEL_4;
      v3 = *((_BYTE *)v17 + 28);
      if ( v3 != 2 )
        goto LABEL_8;
LABEL_11:
      v17[2] = (__int64)v16;
      v18 = (__int64 *)v17[20];
      if ( v18 )
      {
        v4 = *((_BYTE *)v18 + 28);
        if ( v4 == 2 )
          goto LABEL_16;
        while ( 1 )
        {
          if ( v4 == 15 )
            goto LABEL_16;
          while ( 1 )
          {
            v18 = (__int64 *)*v18;
            if ( !v18 )
              goto LABEL_9;
            v4 = *((_BYTE *)v18 + 28);
            if ( v4 != 2 )
              break;
LABEL_16:
            v18[2] = (__int64)v17;
            v19 = (__int64 *)v18[20];
            if ( v19 )
            {
              v5 = *((_BYTE *)v19 + 28);
              if ( v5 != 2 )
                goto LABEL_18;
              while ( 2 )
              {
                v6 = (__int64 *)v19[20];
                v19[2] = (__int64)v18;
                if ( v6 )
                {
                  v7 = *((_BYTE *)v6 + 28);
                  if ( v7 == 2 )
                    goto LABEL_26;
                  while ( 1 )
                  {
                    if ( v7 == 15 )
                      goto LABEL_26;
                    while ( 1 )
                    {
                      v6 = (__int64 *)*v6;
                      if ( !v6 )
                        goto LABEL_19;
                      v7 = *((_BYTE *)v6 + 28);
                      if ( v7 != 2 )
                        break;
LABEL_26:
                      v8 = (__int64 *)v6[20];
                      v6[2] = (__int64)v19;
                      if ( v8 )
                      {
                        v9 = *((_BYTE *)v8 + 28);
                        if ( v9 != 2 )
                          goto LABEL_28;
                        while ( 2 )
                        {
                          v10 = (__int64 *)v8[20];
                          v8[2] = (__int64)v6;
                          if ( v10 )
                          {
                            v11 = *((_BYTE *)v10 + 28);
                            if ( v11 == 15 )
                              goto LABEL_36;
                            while ( 1 )
                            {
                              if ( v11 == 2 )
                                goto LABEL_36;
                              while ( 1 )
                              {
                                v10 = (__int64 *)*v10;
                                if ( !v10 )
                                  goto LABEL_29;
                                v11 = *((_BYTE *)v10 + 28);
                                if ( v11 != 15 )
                                  break;
LABEL_36:
                                v12 = (__int64 *)v10[20];
                                v10[2] = (__int64)v8;
                                if ( v12 )
                                {
                                  v13 = *((_BYTE *)v12 + 28);
                                  if ( v13 != 15 )
                                    goto LABEL_38;
                                  while ( 2 )
                                  {
                                    v14 = (__int64 *)v12[20];
                                    v12[2] = (__int64)v10;
                                    if ( v14 )
                                    {
                                      v15 = *((_BYTE *)v14 + 28);
                                      if ( v15 == 15 )
                                        goto LABEL_46;
                                      while ( 1 )
                                      {
                                        if ( v15 == 2 )
                                          goto LABEL_46;
                                        while ( 1 )
                                        {
                                          v14 = (__int64 *)*v14;
                                          if ( !v14 )
                                            goto LABEL_39;
                                          v15 = *((_BYTE *)v14 + 28);
                                          if ( v15 != 15 )
                                            break;
LABEL_46:
                                          v14[2] = (__int64)v12;
                                          sub_85BD30(v14);
                                        }
                                      }
                                    }
LABEL_39:
                                    v12 = (__int64 *)*v12;
                                    if ( v12 )
                                    {
                                      v13 = *((_BYTE *)v12 + 28);
                                      if ( v13 == 15 )
                                        continue;
LABEL_38:
                                      if ( v13 != 2 )
                                        goto LABEL_39;
                                      continue;
                                    }
                                    break;
                                  }
                                }
                              }
                            }
                          }
LABEL_29:
                          v8 = (__int64 *)*v8;
                          if ( v8 )
                          {
                            v9 = *((_BYTE *)v8 + 28);
                            if ( v9 == 2 )
                              continue;
LABEL_28:
                            if ( v9 != 15 )
                              goto LABEL_29;
                            continue;
                          }
                          break;
                        }
                      }
                    }
                  }
                }
LABEL_19:
                v19 = (__int64 *)*v19;
                if ( v19 )
                {
                  v5 = *((_BYTE *)v19 + 28);
                  if ( v5 == 2 )
                    continue;
LABEL_18:
                  if ( v5 != 15 )
                    goto LABEL_19;
                  continue;
                }
                break;
              }
            }
          }
        }
      }
LABEL_9:
      v17 = (__int64 *)*v17;
      if ( v17 )
        break;
LABEL_4:
      result = (__int64 *)*v16;
      v16 = result;
      if ( !result )
        return result;
    }
    v3 = *((_BYTE *)v17 + 28);
    if ( v3 == 2 )
      goto LABEL_11;
LABEL_8:
    if ( v3 != 15 )
      goto LABEL_9;
    goto LABEL_11;
  }
  return result;
}
