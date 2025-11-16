// Function: sub_31A43D0
// Address: 0x31a43d0
//
__int64 __fastcall sub_31A43D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 *v10; // rbx
  __int64 *v11; // r12
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 *v14; // r13
  __int64 *v15; // r14
  __int64 v16; // r14
  __int64 v17; // r15
  _QWORD *v18; // r12
  _QWORD *v19; // r15
  __int64 *v21; // [rsp+8h] [rbp-A8h]
  __int64 *v22; // [rsp+10h] [rbp-A0h]
  __int64 *v23; // [rsp+18h] [rbp-98h]
  __int64 *v24; // [rsp+20h] [rbp-90h]
  __int64 *v25; // [rsp+28h] [rbp-88h]
  __int64 *v26; // [rsp+30h] [rbp-80h]
  __int64 *v27; // [rsp+38h] [rbp-78h]
  __int64 *v28; // [rsp+40h] [rbp-70h]
  unsigned __int8 v29; // [rsp+4Fh] [rbp-61h]
  __int64 *v30; // [rsp+50h] [rbp-60h]
  __int64 *v31; // [rsp+58h] [rbp-58h]
  __int64 *v32; // [rsp+60h] [rbp-50h]
  __int64 *v33; // [rsp+68h] [rbp-48h]
  __int64 *v34; // [rsp+70h] [rbp-40h]
  __int64 *v35; // [rsp+78h] [rbp-38h]

  v2 = a2;
  v29 = sub_31A4290(a1, a2);
  if ( !v29 )
    return 0;
  v23 = *(__int64 **)(a1 + 16);
  v26 = *(__int64 **)(a1 + 8);
  if ( v26 != v23 )
  {
    while ( 1 )
    {
      v3 = *v26;
      if ( !(unsigned __int8)sub_31A4290(*v26, v2) )
        return 0;
      v22 = *(__int64 **)(v3 + 16);
      if ( *(__int64 **)(v3 + 8) != v22 )
      {
        v25 = *(__int64 **)(v3 + 8);
        do
        {
          v4 = *v25;
          if ( !(unsigned __int8)sub_31A4290(*v25, v2) )
            return 0;
          v21 = *(__int64 **)(v4 + 16);
          if ( *(__int64 **)(v4 + 8) != v21 )
          {
            v24 = *(__int64 **)(v4 + 8);
            do
            {
              v5 = *v24;
              if ( !(unsigned __int8)sub_31A4290(*v24, v2) )
                return 0;
              v27 = *(__int64 **)(v5 + 16);
              if ( *(__int64 **)(v5 + 8) != v27 )
              {
                v31 = *(__int64 **)(v5 + 8);
                v6 = v2;
                while ( 1 )
                {
                  v7 = *v31;
                  if ( !(unsigned __int8)sub_31A4290(*v31, v6) )
                    return 0;
                  v30 = *(__int64 **)(v7 + 16);
                  if ( *(__int64 **)(v7 + 8) != v30 )
                  {
                    v35 = *(__int64 **)(v7 + 8);
                    do
                    {
                      v8 = *v35;
                      if ( !(unsigned __int8)sub_31A4290(*v35, v6) )
                        return 0;
                      v28 = *(__int64 **)(v8 + 16);
                      if ( *(__int64 **)(v8 + 8) != v28 )
                      {
                        v32 = *(__int64 **)(v8 + 8);
                        do
                        {
                          v9 = *v32;
                          if ( !(unsigned __int8)sub_31A4290(*v32, v6) )
                            return 0;
                          v10 = *(__int64 **)(v9 + 8);
                          v11 = *(__int64 **)(v9 + 16);
                          if ( v10 != v11 )
                          {
                            v33 = v11;
                            v12 = v6;
                            while ( 1 )
                            {
                              v13 = *v10;
                              if ( !(unsigned __int8)sub_31A4290(*v10, v12) )
                                return 0;
                              v14 = *(__int64 **)(v13 + 8);
                              v15 = *(__int64 **)(v13 + 16);
                              if ( v14 != v15 )
                              {
                                v34 = v15;
                                v16 = v12;
                                while ( 1 )
                                {
                                  v17 = *v14;
                                  if ( !(unsigned __int8)sub_31A4290(*v14, v16) )
                                    return 0;
                                  v18 = *(_QWORD **)(v17 + 16);
                                  if ( *(_QWORD **)(v17 + 8) != v18 )
                                  {
                                    v19 = *(_QWORD **)(v17 + 8);
                                    while ( (unsigned __int8)sub_31A43D0(*v19, v16) )
                                    {
                                      if ( v18 == ++v19 )
                                        goto LABEL_29;
                                    }
                                    return 0;
                                  }
LABEL_29:
                                  if ( v34 == ++v14 )
                                  {
                                    v12 = v16;
                                    break;
                                  }
                                }
                              }
                              if ( v33 == ++v10 )
                              {
                                v6 = v12;
                                break;
                              }
                            }
                          }
                          ++v32;
                        }
                        while ( v28 != v32 );
                      }
                      ++v35;
                    }
                    while ( v30 != v35 );
                  }
                  if ( v27 == ++v31 )
                  {
                    v2 = v6;
                    break;
                  }
                }
              }
              ++v24;
            }
            while ( v21 != v24 );
          }
          ++v25;
        }
        while ( v22 != v25 );
      }
      if ( v23 == ++v26 )
        return v29;
    }
  }
  return v29;
}
