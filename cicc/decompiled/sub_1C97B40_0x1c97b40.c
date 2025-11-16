// Function: sub_1C97B40
// Address: 0x1c97b40
//
__int64 __fastcall sub_1C97B40(__int64 a1)
{
  int v1; // edx
  __int64 v2; // rax
  char v3; // dl
  int v4; // edx
  __int64 v5; // rax
  char v6; // dl
  int v7; // edx
  __int64 *v8; // r11
  __int64 v9; // rax
  char v10; // dl
  int v11; // edx
  __int64 *v12; // r15
  __int64 v13; // rax
  char v14; // dl
  int v15; // ecx
  __int64 *v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  char v19; // dl
  int v20; // edx
  __int64 *v21; // r12
  __int64 v22; // rsi
  __int64 v23; // r15
  __int64 *v24; // rdx
  __int64 v25; // rcx
  char v26; // al
  int v27; // eax
  __int64 *v28; // rbx
  __int64 v29; // r14
  __int64 v30; // rax
  char v31; // cl
  int v32; // ecx
  __int64 *v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  char v36; // cl
  int v37; // ecx
  __int64 v38; // r13
  __int64 v39; // rcx
  __int64 result; // rax
  char v41; // al
  __int64 *v42; // [rsp-A0h] [rbp-A0h]
  __int64 v43; // [rsp-98h] [rbp-98h]
  __int64 v44; // [rsp-90h] [rbp-90h]
  __int64 v45; // [rsp-88h] [rbp-88h]
  __int64 *v46; // [rsp-80h] [rbp-80h]
  __int64 *v47; // [rsp-78h] [rbp-78h]
  __int64 *v48; // [rsp-70h] [rbp-70h]
  __int64 v49; // [rsp-68h] [rbp-68h]
  __int64 v50; // [rsp-58h] [rbp-58h]
  __int64 *v51; // [rsp-50h] [rbp-50h]
  __int64 v52; // [rsp-48h] [rbp-48h]
  __int64 *v53; // [rsp-40h] [rbp-40h]

  v1 = *(_DWORD *)(a1 + 12);
  if ( !v1 )
    return 0;
  v47 = *(__int64 **)(a1 + 16);
  v43 = (__int64)&v47[(unsigned int)(v1 - 1) + 1];
  do
  {
    v2 = *v47;
    v3 = *(_BYTE *)(*v47 + 8);
    if ( v3 == 15 )
      return 1;
    if ( v3 == 13 )
    {
      v4 = *(_DWORD *)(v2 + 12);
      if ( v4 )
      {
        v48 = *(__int64 **)(v2 + 16);
        v44 = (__int64)&v48[(unsigned int)(v4 - 1) + 1];
        do
        {
          v5 = *v48;
          v6 = *(_BYTE *)(*v48 + 8);
          if ( v6 == 15 )
            return 1;
          if ( v6 == 13 )
          {
            v7 = *(_DWORD *)(v5 + 12);
            if ( v7 )
            {
              v8 = *(__int64 **)(v5 + 16);
              v45 = (__int64)&v8[(unsigned int)(v7 - 1) + 1];
              do
              {
                v9 = *v8;
                v10 = *(_BYTE *)(*v8 + 8);
                if ( v10 == 15 )
                  return 1;
                if ( v10 == 13 )
                {
                  v11 = *(_DWORD *)(v9 + 12);
                  if ( v11 )
                  {
                    v42 = v8;
                    v12 = *(__int64 **)(v9 + 16);
                    v49 = (__int64)&v12[(unsigned int)(v11 - 1) + 1];
                    while ( 1 )
                    {
                      v13 = *v12;
                      v14 = *(_BYTE *)(*v12 + 8);
                      if ( v14 == 15 )
                        return 1;
                      if ( v14 == 13 )
                      {
                        v15 = *(_DWORD *)(v13 + 12);
                        if ( v15 )
                        {
                          v46 = v12;
                          v16 = *(__int64 **)(v13 + 16);
                          v17 = (__int64)&v16[(unsigned int)(v15 - 1) + 1];
                          while ( 1 )
                          {
                            v18 = *v16;
                            v19 = *(_BYTE *)(*v16 + 8);
                            if ( v19 == 15 )
                              return 1;
                            if ( v19 == 13 )
                            {
                              v20 = *(_DWORD *)(v18 + 12);
                              if ( v20 )
                              {
                                v21 = *(__int64 **)(v18 + 16);
                                v22 = v17;
                                v23 = (__int64)&v21[(unsigned int)(v20 - 1) + 1];
                                v24 = v16;
                                while ( 1 )
                                {
                                  v25 = *v21;
                                  v26 = *(_BYTE *)(*v21 + 8);
                                  if ( v26 == 15 )
                                    return 1;
                                  if ( v26 == 13 )
                                  {
                                    v27 = *(_DWORD *)(v25 + 12);
                                    if ( v27 )
                                    {
                                      v28 = *(__int64 **)(v25 + 16);
                                      v29 = (__int64)&v28[(unsigned int)(v27 - 1) + 1];
                                      do
                                      {
                                        v30 = *v28;
                                        v31 = *(_BYTE *)(*v28 + 8);
                                        if ( v31 == 15 )
                                          return 1;
                                        if ( v31 == 13 )
                                        {
                                          v32 = *(_DWORD *)(v30 + 12);
                                          if ( v32 )
                                          {
                                            v33 = *(__int64 **)(v30 + 16);
                                            v34 = (__int64)&v33[(unsigned int)(v32 - 1) + 1];
                                            do
                                            {
                                              v35 = *v33;
                                              v36 = *(_BYTE *)(*v33 + 8);
                                              if ( v36 == 15 )
                                                return 1;
                                              if ( v36 == 13 )
                                              {
                                                v37 = *(_DWORD *)(v35 + 12);
                                                if ( v37 )
                                                {
                                                  v38 = *(_QWORD *)(v35 + 16);
                                                  v39 = v38 + 8LL * (unsigned int)(v37 - 1) + 8;
                                                  do
                                                  {
                                                    v41 = *(_BYTE *)(*(_QWORD *)v38 + 8LL);
                                                    if ( v41 == 15 )
                                                      return 1;
                                                    if ( v41 == 13 )
                                                    {
                                                      v50 = v39;
                                                      v51 = v33;
                                                      v52 = v34;
                                                      v53 = v24;
                                                      result = sub_1C97B40();
                                                      v24 = v53;
                                                      v34 = v52;
                                                      v33 = v51;
                                                      v39 = v50;
                                                      if ( (_BYTE)result )
                                                        return result;
                                                    }
                                                    v38 += 8;
                                                  }
                                                  while ( v39 != v38 );
                                                }
                                              }
                                              ++v33;
                                            }
                                            while ( (__int64 *)v34 != v33 );
                                          }
                                        }
                                        ++v28;
                                      }
                                      while ( (__int64 *)v29 != v28 );
                                    }
                                  }
                                  if ( (__int64 *)v23 == ++v21 )
                                  {
                                    v16 = v24;
                                    v17 = v22;
                                    break;
                                  }
                                }
                              }
                            }
                            if ( ++v16 == (__int64 *)v17 )
                            {
                              v12 = v46;
                              break;
                            }
                          }
                        }
                      }
                      if ( (__int64 *)v49 == ++v12 )
                      {
                        v8 = v42;
                        break;
                      }
                    }
                  }
                }
                ++v8;
              }
              while ( (__int64 *)v45 != v8 );
            }
          }
          ++v48;
        }
        while ( v48 != (__int64 *)v44 );
      }
    }
    ++v47;
  }
  while ( v47 != (__int64 *)v43 );
  return 0;
}
