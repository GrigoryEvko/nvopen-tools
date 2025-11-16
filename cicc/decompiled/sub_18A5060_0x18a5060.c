// Function: sub_18A5060
// Address: 0x18a5060
//
__int64 __fastcall sub_18A5060(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  _QWORD *v3; // r15
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // esi
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // esi
  _QWORD *v14; // r15
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned int v19; // eax
  _QWORD *v20; // r13
  _QWORD *v21; // rbx
  __int64 v22; // r15
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r12
  unsigned int v28; // eax
  _QWORD *v29; // r9
  _QWORD *v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // ecx
  __int64 v36; // rax
  __int64 v38; // rax
  _QWORD *v39; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  _QWORD *v41; // [rsp+10h] [rbp-70h]
  _QWORD *v42; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+20h] [rbp-60h]
  _QWORD *v44; // [rsp+28h] [rbp-58h]
  _QWORD *v45; // [rsp+30h] [rbp-50h]
  __int64 v46; // [rsp+38h] [rbp-48h]
  _QWORD *v47; // [rsp+40h] [rbp-40h]
  _QWORD *v48; // [rsp+48h] [rbp-38h]

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 104);
  v48 = (_QWORD *)(v2 + 48);
  if ( *(_QWORD *)(v2 + 64) != v2 + 48 )
  {
    v3 = *(_QWORD **)(v2 + 64);
    v4 = 0;
    while ( 1 )
    {
      v5 = v3[23];
      if ( v3[17] )
      {
        v6 = v3[15];
        if ( v5 )
        {
          v7 = v3[21];
          v8 = *(_DWORD *)(v7 + 32);
          if ( *(_DWORD *)(v6 + 32) >= v8
            && (*(_DWORD *)(v6 + 32) != v8 || *(_DWORD *)(v6 + 36) >= *(_DWORD *)(v7 + 36)) )
          {
            goto LABEL_7;
          }
        }
        v4 += *(_QWORD *)(v6 + 40);
      }
      else if ( v5 )
      {
        v7 = v3[21];
LABEL_7:
        v9 = *(_QWORD **)(v7 + 64);
        v47 = (_QWORD *)(v7 + 48);
        if ( v9 != (_QWORD *)(v7 + 48) )
        {
          v46 = 0;
          v44 = v3;
          v43 = v4;
          do
          {
            v10 = v9[23];
            if ( v9[17] )
            {
              v11 = v9[15];
              if ( v10 )
              {
                v12 = v9[21];
                v13 = *(_DWORD *)(v12 + 32);
                if ( *(_DWORD *)(v11 + 32) >= v13
                  && (*(_DWORD *)(v11 + 32) != v13 || *(_DWORD *)(v11 + 36) >= *(_DWORD *)(v12 + 36)) )
                {
                  goto LABEL_13;
                }
              }
              v46 += *(_QWORD *)(v11 + 40);
            }
            else if ( v10 )
            {
              v12 = v9[21];
LABEL_13:
              v14 = *(_QWORD **)(v12 + 64);
              v45 = (_QWORD *)(v12 + 48);
              if ( v14 == (_QWORD *)(v12 + 48) )
                goto LABEL_47;
              v42 = v9;
              v15 = 0;
              while ( 2 )
              {
                v16 = v14[23];
                if ( v14[17] )
                {
                  v17 = v14[15];
                  if ( !v16
                    || (v18 = v14[21], v19 = *(_DWORD *)(v18 + 32), *(_DWORD *)(v17 + 32) < v19)
                    || *(_DWORD *)(v17 + 32) == v19 && *(_DWORD *)(v17 + 36) < *(_DWORD *)(v18 + 36) )
                  {
                    v15 += *(_QWORD *)(v17 + 40);
                    goto LABEL_45;
                  }
                }
                else
                {
                  if ( !v16 )
                    goto LABEL_45;
                  v18 = v14[21];
                }
                v20 = *(_QWORD **)(v18 + 64);
                v21 = (_QWORD *)(v18 + 48);
                if ( v20 != v21 )
                {
                  v41 = v14;
                  v40 = v15;
                  v22 = 0;
                  v23 = v20;
                  v24 = v21;
                  do
                  {
                    v25 = v23[23];
                    if ( v23[17] )
                    {
                      v26 = v23[15];
                      if ( v25 )
                      {
                        v27 = v23[21];
                        v28 = *(_DWORD *)(v27 + 32);
                        if ( *(_DWORD *)(v26 + 32) >= v28
                          && (*(_DWORD *)(v26 + 32) != v28 || *(_DWORD *)(v26 + 36) >= *(_DWORD *)(v27 + 36)) )
                        {
                          goto LABEL_25;
                        }
                      }
                      v22 += *(_QWORD *)(v26 + 40);
                    }
                    else if ( v25 )
                    {
                      v27 = v23[21];
LABEL_25:
                      v29 = *(_QWORD **)(v27 + 64);
                      v30 = (_QWORD *)(v27 + 48);
                      if ( v29 == v30 )
                        goto LABEL_43;
                      v31 = 0;
                      while ( 2 )
                      {
                        v32 = v29[23];
                        if ( v29[17] )
                        {
                          v33 = v29[15];
                          if ( !v32
                            || (v34 = v29[21], v35 = *(_DWORD *)(v34 + 32), *(_DWORD *)(v33 + 32) < v35)
                            || *(_DWORD *)(v33 + 32) == v35 && *(_DWORD *)(v33 + 36) < *(_DWORD *)(v34 + 36) )
                          {
                            v31 += *(_QWORD *)(v33 + 40);
LABEL_32:
                            v29 = (_QWORD *)sub_220EF30(v29);
                            if ( v30 == v29 )
                            {
                              v22 += v31;
                              goto LABEL_43;
                            }
                            continue;
                          }
                        }
                        else if ( !v32 )
                        {
                          goto LABEL_32;
                        }
                        break;
                      }
                      v39 = v29;
                      v36 = sub_18A5060(v29 + 8);
                      v29 = v39;
                      v31 += v36;
                      goto LABEL_32;
                    }
LABEL_43:
                    v23 = (_QWORD *)sub_220EF30(v23);
                  }
                  while ( v24 != v23 );
                  v38 = v22;
                  v14 = v41;
                  v15 = v38 + v40;
                }
LABEL_45:
                v14 = (_QWORD *)sub_220EF30(v14);
                if ( v45 == v14 )
                {
                  v46 += v15;
                  v9 = v42;
                  break;
                }
                continue;
              }
            }
LABEL_47:
            v9 = (_QWORD *)sub_220EF30(v9);
          }
          while ( v47 != v9 );
          v3 = v44;
          v4 = v46 + v43;
        }
      }
      v3 = (_QWORD *)sub_220EF30(v3);
      if ( v48 == v3 )
        return v4;
    }
  }
  return v1;
}
