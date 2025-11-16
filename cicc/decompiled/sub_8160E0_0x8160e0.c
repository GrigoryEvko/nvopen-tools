// Function: sub_8160E0
// Address: 0x8160e0
//
__int64 __fastcall sub_8160E0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rbx
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 i; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  sub_816050(*(_QWORD *)(a1 + 104));
  result = *(_QWORD *)(a1 + 168);
  for ( i = result; result; i = result )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
    {
      v2 = *(_QWORD *)(i + 128);
      sub_816050(*(_QWORD *)(v2 + 104));
      v24 = *(_QWORD *)(v2 + 168);
      if ( v24 )
      {
        v3 = *(_QWORD *)(v2 + 168);
        if ( (*(_BYTE *)(v24 + 124) & 1) == 0 )
        {
LABEL_9:
          v4 = *(_QWORD *)(v3 + 128);
          sub_816050(*(_QWORD *)(v4 + 104));
          v5 = *(_QWORD *)(v4 + 168);
          if ( v5 )
          {
            if ( (*(_BYTE *)(v5 + 124) & 1) == 0 )
            {
LABEL_13:
              v6 = *(_QWORD *)(v5 + 128);
              sub_816050(*(_QWORD *)(v6 + 104));
              v7 = *(_QWORD *)(v6 + 168);
              if ( v7 )
              {
                if ( (*(_BYTE *)(v7 + 124) & 1) == 0 )
                {
LABEL_17:
                  v8 = *(_QWORD *)(v7 + 128);
                  sub_816050(*(_QWORD *)(v8 + 104));
                  v9 = *(_QWORD *)(v8 + 168);
                  if ( v9 )
                  {
                    if ( (*(_BYTE *)(v9 + 124) & 1) == 0 )
                    {
LABEL_21:
                      v10 = *(_QWORD *)(v9 + 128);
                      sub_816050(*(_QWORD *)(v10 + 104));
                      v11 = *(_QWORD *)(v10 + 168);
                      if ( v11 )
                      {
                        if ( (*(_BYTE *)(v11 + 124) & 1) == 0 )
                        {
LABEL_25:
                          v12 = *(_QWORD *)(v11 + 128);
                          sub_816050(*(_QWORD *)(v12 + 104));
                          v13 = *(_QWORD *)(v12 + 168);
                          if ( v13 )
                          {
                            if ( (*(_BYTE *)(v13 + 124) & 1) == 0 )
                            {
LABEL_29:
                              v14 = *(_QWORD *)(v13 + 128);
                              v20 = v13;
                              sub_816050(*(_QWORD *)(v14 + 104));
                              v15 = *(_QWORD *)(v14 + 168);
                              v13 = v20;
                              if ( v15 )
                              {
                                if ( (*(_BYTE *)(v15 + 124) & 1) == 0 )
                                {
LABEL_33:
                                  v16 = *(_QWORD *)(v15 + 128);
                                  v18 = v13;
                                  v21 = v15;
                                  sub_816050(*(_QWORD *)(v16 + 104));
                                  v17 = *(_QWORD *)(v16 + 168);
                                  v15 = v21;
                                  v13 = v18;
                                  if ( v17 )
                                  {
                                    if ( (*(_BYTE *)(v17 + 124) & 1) == 0 )
                                      goto LABEL_37;
                                    while ( 1 )
                                    {
                                      v17 = *(_QWORD *)(v17 + 112);
                                      if ( !v17 )
                                        break;
                                      if ( (*(_BYTE *)(v17 + 124) & 1) == 0 )
                                      {
LABEL_37:
                                        v19 = v15;
                                        v22 = v13;
                                        sub_8160E0(*(_QWORD *)(v17 + 128));
                                        v15 = v19;
                                        v13 = v22;
                                      }
                                    }
                                  }
                                }
                                while ( 1 )
                                {
                                  v15 = *(_QWORD *)(v15 + 112);
                                  if ( !v15 )
                                    break;
                                  if ( (*(_BYTE *)(v15 + 124) & 1) == 0 )
                                    goto LABEL_33;
                                }
                              }
                            }
                            while ( 1 )
                            {
                              v13 = *(_QWORD *)(v13 + 112);
                              if ( !v13 )
                                break;
                              if ( (*(_BYTE *)(v13 + 124) & 1) == 0 )
                                goto LABEL_29;
                            }
                          }
                        }
                        while ( 1 )
                        {
                          v11 = *(_QWORD *)(v11 + 112);
                          if ( !v11 )
                            break;
                          if ( (*(_BYTE *)(v11 + 124) & 1) == 0 )
                            goto LABEL_25;
                        }
                      }
                    }
                    while ( 1 )
                    {
                      v9 = *(_QWORD *)(v9 + 112);
                      if ( !v9 )
                        break;
                      if ( (*(_BYTE *)(v9 + 124) & 1) == 0 )
                        goto LABEL_21;
                    }
                  }
                }
                while ( 1 )
                {
                  v7 = *(_QWORD *)(v7 + 112);
                  if ( !v7 )
                    break;
                  if ( (*(_BYTE *)(v7 + 124) & 1) == 0 )
                    goto LABEL_17;
                }
              }
            }
            while ( 1 )
            {
              v5 = *(_QWORD *)(v5 + 112);
              if ( !v5 )
                break;
              if ( (*(_BYTE *)(v5 + 124) & 1) == 0 )
                goto LABEL_13;
            }
          }
        }
        while ( 1 )
        {
          v24 = *(_QWORD *)(v24 + 112);
          if ( !v24 )
            break;
          v3 = v24;
          if ( (*(_BYTE *)(v24 + 124) & 1) == 0 )
            goto LABEL_9;
        }
      }
    }
    result = *(_QWORD *)(i + 112);
  }
  return result;
}
