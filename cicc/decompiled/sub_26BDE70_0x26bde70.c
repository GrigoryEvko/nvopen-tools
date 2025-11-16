// Function: sub_26BDE70
// Address: 0x26bde70
//
__int64 __fastcall sub_26BDE70(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 k; // rbx
  __int64 v5; // rax
  __int64 m; // rbx
  __int64 v7; // rax
  __int64 n; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 ii; // r15
  __int64 v21; // r8
  __int64 v22; // r12
  __int64 v23; // r15
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-128h]
  __int64 v29; // [rsp+10h] [rbp-120h]
  __int64 v30; // [rsp+18h] [rbp-118h]
  __int64 v31; // [rsp+20h] [rbp-110h]
  __int64 v32; // [rsp+28h] [rbp-108h]
  __int64 v33; // [rsp+30h] [rbp-100h]
  __int64 v34; // [rsp+38h] [rbp-F8h]
  __int64 v35; // [rsp+40h] [rbp-F0h]
  __int64 v36; // [rsp+48h] [rbp-E8h]
  __int64 v37; // [rsp+50h] [rbp-E0h]
  __int64 v38; // [rsp+58h] [rbp-D8h]
  __int64 v39; // [rsp+60h] [rbp-D0h]
  __int64 v40; // [rsp+68h] [rbp-C8h]
  __int64 v41; // [rsp+70h] [rbp-C0h]
  __int64 v42; // [rsp+78h] [rbp-B8h]
  __int64 v43; // [rsp+88h] [rbp-A8h]
  __int64 v44; // [rsp+98h] [rbp-98h]
  __int64 v45; // [rsp+A8h] [rbp-88h]
  __int64 v46; // [rsp+D0h] [rbp-60h]
  __int64 v47; // [rsp+D8h] [rbp-58h]
  __int64 v48; // [rsp+E0h] [rbp-50h]
  __int64 j; // [rsp+E8h] [rbp-48h]
  __int64 i; // [rsp+F0h] [rbp-40h]
  __int64 v51; // [rsp+F8h] [rbp-38h]

  result = a1 + 128;
  v2 = *(_QWORD *)(a1 + 144);
  *(_DWORD *)(a1 + 48) |= 2u;
  v51 = v2;
  if ( v2 != a1 + 128 )
  {
    do
    {
      for ( i = *(_QWORD *)(v51 + 64); v51 + 48 != i; i = sub_220EEE0(i) )
      {
        v3 = *(_QWORD *)(i + 192);
        *(_DWORD *)(i + 96) |= 2u;
        for ( j = v3; i + 176 != j; j = sub_220EEE0(j) )
        {
          for ( k = *(_QWORD *)(j + 64); j + 48 != k; k = sub_220EEE0(k) )
          {
            v5 = *(_QWORD *)(k + 192);
            *(_DWORD *)(k + 96) |= 2u;
            v45 = k + 176;
            v48 = v5;
            if ( v5 != k + 176 )
            {
              v38 = k;
              do
              {
                for ( m = *(_QWORD *)(v48 + 64); v48 + 48 != m; m = sub_220EEE0(m) )
                {
                  v7 = *(_QWORD *)(m + 192);
                  *(_DWORD *)(m + 96) |= 2u;
                  v44 = m + 176;
                  v47 = v7;
                  if ( v7 != m + 176 )
                  {
                    v37 = m;
                    do
                    {
                      for ( n = *(_QWORD *)(v47 + 64); v47 + 48 != n; n = sub_220EEE0(n) )
                      {
                        v9 = *(_QWORD *)(n + 192);
                        *(_DWORD *)(n + 96) |= 2u;
                        v43 = n + 176;
                        v46 = v9;
                        if ( v9 != n + 176 )
                        {
                          v36 = n;
                          do
                          {
                            if ( *(_QWORD *)(v46 + 64) != v46 + 48 )
                            {
                              v10 = *(_QWORD *)(v46 + 64);
                              do
                              {
                                v11 = *(_QWORD *)(v10 + 192);
                                *(_DWORD *)(v10 + 96) |= 2u;
                                v42 = v10 + 176;
                                if ( v11 != v10 + 176 )
                                {
                                  v35 = v10;
                                  v12 = v11;
                                  do
                                  {
                                    v13 = *(_QWORD *)(v12 + 64);
                                    v41 = v12 + 48;
                                    if ( v13 != v12 + 48 )
                                    {
                                      v34 = v12;
                                      do
                                      {
                                        v14 = *(_QWORD *)(v13 + 192);
                                        *(_DWORD *)(v13 + 96) |= 2u;
                                        v40 = v13 + 176;
                                        if ( v14 != v13 + 176 )
                                        {
                                          v33 = v13;
                                          do
                                          {
                                            v15 = *(_QWORD *)(v14 + 64);
                                            v39 = v14 + 48;
                                            if ( v15 != v14 + 48 )
                                            {
                                              v32 = v14;
                                              do
                                              {
                                                v16 = *(_QWORD *)(v15 + 192);
                                                v17 = v15 + 176;
                                                *(_DWORD *)(v15 + 96) |= 2u;
                                                if ( v16 != v15 + 176 )
                                                {
                                                  v31 = v15;
                                                  v18 = v16;
                                                  do
                                                  {
                                                    v19 = *(_QWORD *)(v18 + 64);
                                                    for ( ii = v18 + 48; ii != v19; v19 = sub_220EEE0(v19) )
                                                    {
                                                      v21 = *(_QWORD *)(v19 + 192);
                                                      v22 = v19 + 176;
                                                      *(_DWORD *)(v19 + 96) |= 2u;
                                                      if ( v21 != v19 + 176 )
                                                      {
                                                        do
                                                        {
                                                          if ( *(_QWORD *)(v21 + 64) != v21 + 48 )
                                                          {
                                                            v30 = ii;
                                                            v23 = v19;
                                                            v24 = v17;
                                                            v25 = v21 + 48;
                                                            v29 = v22;
                                                            v26 = *(_QWORD *)(v21 + 64);
                                                            do
                                                            {
                                                              v28 = v21;
                                                              sub_26BDE70(v26 + 48);
                                                              v27 = sub_220EEE0(v26);
                                                              v21 = v28;
                                                              v26 = v27;
                                                            }
                                                            while ( v25 != v27 );
                                                            v17 = v24;
                                                            v22 = v29;
                                                            v19 = v23;
                                                            ii = v30;
                                                          }
                                                          v21 = sub_220EEE0(v21);
                                                        }
                                                        while ( v22 != v21 );
                                                      }
                                                    }
                                                    v18 = sub_220EEE0(v18);
                                                  }
                                                  while ( v17 != v18 );
                                                  v15 = v31;
                                                }
                                                v15 = sub_220EEE0(v15);
                                              }
                                              while ( v39 != v15 );
                                              v14 = v32;
                                            }
                                            v14 = sub_220EEE0(v14);
                                          }
                                          while ( v40 != v14 );
                                          v13 = v33;
                                        }
                                        v13 = sub_220EEE0(v13);
                                      }
                                      while ( v41 != v13 );
                                      v12 = v34;
                                    }
                                    v12 = sub_220EEE0(v12);
                                  }
                                  while ( v42 != v12 );
                                  v10 = v35;
                                }
                                v10 = sub_220EEE0(v10);
                              }
                              while ( v46 + 48 != v10 );
                            }
                            v46 = sub_220EEE0(v46);
                          }
                          while ( v43 != v46 );
                          n = v36;
                        }
                      }
                      v47 = sub_220EEE0(v47);
                    }
                    while ( v44 != v47 );
                    m = v37;
                  }
                }
                v48 = sub_220EEE0(v48);
              }
              while ( v45 != v48 );
              k = v38;
            }
          }
        }
      }
      result = sub_220EEE0(v51);
      v51 = result;
    }
    while ( a1 + 128 != result );
  }
  return result;
}
