// Function: sub_1039530
// Address: 0x1039530
//
__int64 __fastcall sub_1039530(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 j; // r12
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 k; // r13
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+20h] [rbp-80h]
  __int64 v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h]
  __int64 v28; // [rsp+40h] [rbp-60h]
  __int64 v29; // [rsp+48h] [rbp-58h]
  __int64 v30; // [rsp+50h] [rbp-50h]
  __int64 v31; // [rsp+58h] [rbp-48h]
  __int64 i; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]

  if ( (*(_BYTE *)a2 & 4) != 0 )
    *(_BYTE *)a2 = *(_BYTE *)a2 & 0xFA | 1;
  result = a2 + 40;
  v31 = *(_QWORD *)(a2 + 56);
  if ( a2 + 40 != v31 )
  {
    do
    {
      v3 = *(_QWORD *)(v31 + 40);
      if ( (*(_BYTE *)v3 & 4) != 0 )
        *(_BYTE *)v3 = *(_BYTE *)v3 & 0xFA | 1;
      v27 = v3 + 40;
      v30 = *(_QWORD *)(v3 + 56);
      if ( v30 != v3 + 40 )
      {
        do
        {
          v4 = *(_QWORD *)(v30 + 40);
          if ( (*(_BYTE *)v4 & 4) != 0 )
            *(_BYTE *)v4 = *(_BYTE *)v4 & 0xFA | 1;
          v26 = v4 + 40;
          v29 = *(_QWORD *)(v4 + 56);
          if ( v29 != v4 + 40 )
          {
            do
            {
              v5 = *(_QWORD *)(v29 + 40);
              if ( (*(_BYTE *)v5 & 4) != 0 )
                *(_BYTE *)v5 = *(_BYTE *)v5 & 0xFA | 1;
              v25 = v5 + 40;
              v28 = *(_QWORD *)(v5 + 56);
              if ( v28 != v5 + 40 )
              {
                do
                {
                  v6 = *(_QWORD *)(v28 + 40);
                  if ( (*(_BYTE *)v6 & 4) != 0 )
                    *(_BYTE *)v6 = *(_BYTE *)v6 & 0xFA | 1;
                  v7 = *(_QWORD *)(v6 + 56);
                  for ( i = v6 + 40; i != v7; v7 = sub_220EEE0(v7) )
                  {
                    v8 = *(_QWORD *)(v7 + 40);
                    if ( (*(_BYTE *)v8 & 4) != 0 )
                      *(_BYTE *)v8 = *(_BYTE *)v8 & 0xFA | 1;
                    v9 = *(_QWORD *)(v8 + 56);
                    v10 = v8 + 40;
                    if ( v9 != v10 )
                    {
                      v33 = v7;
                      v11 = v9;
                      do
                      {
                        v12 = *(_QWORD *)(v11 + 40);
                        if ( (*(_BYTE *)v12 & 4) != 0 )
                          *(_BYTE *)v12 = *(_BYTE *)v12 & 0xFA | 1;
                        v13 = *(_QWORD *)(v12 + 56);
                        for ( j = v12 + 40; j != v13; v13 = sub_220EEE0(v13) )
                        {
                          v15 = *(_QWORD *)(v13 + 40);
                          if ( (*(_BYTE *)v15 & 4) != 0 )
                            *(_BYTE *)v15 = *(_BYTE *)v15 & 0xFA | 1;
                          v16 = *(_QWORD *)(v15 + 56);
                          for ( k = v15 + 40; k != v16; v16 = sub_220EEE0(v16) )
                          {
                            v18 = *(_QWORD *)(v16 + 40);
                            if ( (*(_BYTE *)v18 & 4) != 0 )
                              *(_BYTE *)v18 = *(_BYTE *)v18 & 0xFA | 1;
                            v19 = *(_QWORD *)(v18 + 56);
                            v20 = v18 + 40;
                            if ( v19 != v18 + 40 )
                            {
                              do
                              {
                                v22 = v20;
                                v23 = v16;
                                v24 = v19;
                                sub_1039530(a1, *(_QWORD *)(v19 + 40));
                                v21 = sub_220EEE0(v24);
                                v20 = v22;
                                v16 = v23;
                                v19 = v21;
                              }
                              while ( v22 != v21 );
                            }
                          }
                        }
                        v11 = sub_220EEE0(v11);
                      }
                      while ( v10 != v11 );
                      v7 = v33;
                    }
                  }
                  v28 = sub_220EEE0(v28);
                }
                while ( v25 != v28 );
              }
              v29 = sub_220EEE0(v29);
            }
            while ( v26 != v29 );
          }
          v30 = sub_220EEE0(v30);
        }
        while ( v27 != v30 );
      }
      result = sub_220EEE0(v31);
      v31 = result;
    }
    while ( a2 + 40 != result );
  }
  return result;
}
