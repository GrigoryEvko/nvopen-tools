// Function: sub_2B40F30
// Address: 0x2b40f30
//
__int64 __fastcall sub_2B40F30(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // rbp
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rcx
  _DWORD v14[4]; // [rsp-38h] [rbp-38h] BYREF
  _QWORD *v15; // [rsp-28h] [rbp-28h]
  int v16; // [rsp-20h] [rbp-20h]
  _QWORD *v17; // [rsp-18h] [rbp-18h]
  __int64 v18; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)a1 != 85 )
    goto LABEL_23;
  v4 = *(_QWORD *)(a1 - 32);
  if ( !v4 )
    goto LABEL_23;
  if ( *(_BYTE *)v4
    || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80)
    || *(_DWORD *)(v4 + 36) != 237
    || (v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) == 0 )
  {
LABEL_5:
    if ( v4 )
    {
      if ( *(_BYTE *)v4 )
        goto LABEL_64;
      if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80) )
        goto LABEL_64;
      if ( *(_DWORD *)(v4 + 36) != 248 )
        goto LABEL_64;
      v9 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      if ( !v9 )
        goto LABEL_64;
      *a2 = v9;
      if ( *(_BYTE *)a1 == 85 )
      {
        v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
        if ( v6 )
          goto LABEL_28;
        v4 = *(_QWORD *)(a1 - 32);
        if ( v4 )
        {
LABEL_64:
          if ( *(_BYTE *)v4 )
            goto LABEL_61;
          if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80) )
            goto LABEL_61;
          if ( *(_DWORD *)(v4 + 36) != 235 )
            goto LABEL_61;
          v10 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          if ( !v10 )
            goto LABEL_61;
          *a2 = v10;
          if ( *(_BYTE *)a1 == 85 )
          {
            v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
            if ( v6 )
              goto LABEL_28;
            v4 = *(_QWORD *)(a1 - 32);
            if ( v4 )
            {
LABEL_61:
              if ( *(_BYTE *)v4 )
                goto LABEL_62;
              if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80) )
                goto LABEL_62;
              if ( *(_DWORD *)(v4 + 36) != 246 )
                goto LABEL_62;
              v11 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
              if ( !v11 )
                goto LABEL_62;
              *a2 = v11;
              if ( *(_BYTE *)a1 == 85 )
              {
                v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
                if ( v6 )
                  goto LABEL_28;
                v4 = *(_QWORD *)(a1 - 32);
                if ( v4 )
                {
LABEL_62:
                  if ( *(_BYTE *)v4 )
                    goto LABEL_63;
                  if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80) )
                    goto LABEL_63;
                  if ( *(_DWORD *)(v4 + 36) != 329 )
                    goto LABEL_63;
                  v12 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
                  if ( !v12 )
                    goto LABEL_63;
                  *a2 = v12;
                  if ( *(_BYTE *)a1 == 85 )
                  {
                    v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
                    if ( v6 )
                      goto LABEL_28;
                    v4 = *(_QWORD *)(a1 - 32);
                    if ( v4 )
                    {
LABEL_63:
                      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a1 + 80) && *(_DWORD *)(v4 + 36) == 330 )
                      {
                        v13 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
                        if ( v13 )
                        {
                          *a2 = v13;
                          if ( *(_BYTE *)a1 != 85 )
                            goto LABEL_23;
                          v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
                          if ( v6 )
                            goto LABEL_28;
                          v4 = *(_QWORD *)(a1 - 32);
                        }
                      }
                      if ( v4 )
                      {
                        if ( !*(_BYTE *)v4
                          && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a1 + 80)
                          && *(_DWORD *)(v4 + 36) == 365 )
                        {
                          v5 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
                          if ( v5 )
                          {
                            *a2 = v5;
                            if ( *(_BYTE *)a1 == 85 )
                            {
                              v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
                              if ( v6 )
                                goto LABEL_28;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_23:
    v18 = v3;
    v15 = a2;
    v14[0] = 366;
    v14[2] = 0;
    v16 = 1;
    v17 = a3;
    return sub_2B40EB0((__int64)v14, a1);
  }
  *a2 = v8;
  if ( *(_BYTE *)a1 != 85 )
    goto LABEL_23;
  v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( !v6 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    goto LABEL_5;
  }
LABEL_28:
  *a3 = v6;
  return 1;
}
