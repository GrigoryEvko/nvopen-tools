// Function: sub_3153D30
// Address: 0x3153d30
//
void __fastcall sub_3153D30(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  _QWORD *v6; // rax
  unsigned __int64 v7; // r9
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rdx
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // r15
  unsigned __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  _QWORD *v20; // r14
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // r13
  unsigned __int64 v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r12
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rbx
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rbx
  unsigned __int64 v37; // rdi
  _QWORD *v38; // rax
  __int64 v39; // rax
  _QWORD *v40; // rbx
  unsigned __int64 v41; // rdi
  _QWORD *v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // [rsp+8h] [rbp-58h]
  _QWORD *v45; // [rsp+8h] [rbp-58h]
  _QWORD *v46; // [rsp+10h] [rbp-50h]
  _QWORD *v47; // [rsp+18h] [rbp-48h]
  _QWORD *v48; // [rsp+20h] [rbp-40h]
  _QWORD *v49; // [rsp+28h] [rbp-38h]
  _QWORD *v50; // [rsp+28h] [rbp-38h]
  _QWORD *v51; // [rsp+28h] [rbp-38h]
  _QWORD *v52; // [rsp+28h] [rbp-38h]

  v46 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v48 = (_QWORD *)v46[3];
      if ( v48 )
      {
        while ( 1 )
        {
          v47 = (_QWORD *)v48[3];
          if ( v47 )
          {
            while ( 1 )
            {
              v1 = (_QWORD *)v47[3];
              if ( v1 )
              {
                while ( 1 )
                {
                  v2 = (_QWORD *)v1[3];
                  if ( v2 )
                  {
                    while ( 1 )
                    {
                      v3 = (_QWORD *)v2[3];
                      if ( v3 )
                      {
                        while ( 1 )
                        {
                          v4 = (_QWORD *)v3[3];
                          if ( v4 )
                          {
                            do
                            {
                              v5 = (_QWORD *)v4[3];
                              if ( v5 )
                              {
                                do
                                {
                                  v6 = (_QWORD *)v5[3];
                                  if ( v6 )
                                  {
                                    do
                                    {
                                      v49 = v6;
                                      sub_3153D30(v6[3]);
                                      v44 = (_QWORD *)v49[2];
                                      sub_31541A0(v49[28]);
                                      v7 = (unsigned __int64)v49;
                                      v8 = v44;
                                      v9 = v49[8];
                                      if ( (_QWORD *)v9 != v49 + 10 )
                                      {
                                        v45 = v49;
                                        v50 = v8;
                                        _libc_free(v9);
                                        v7 = (unsigned __int64)v45;
                                        v8 = v50;
                                      }
                                      v10 = *(_QWORD **)(v7 + 48);
                                      if ( v10 )
                                        *v10 = *(_QWORD *)(v7 + 40);
                                      v11 = *(_QWORD *)(v7 + 40);
                                      if ( v11 )
                                        *(_QWORD *)(v11 + 8) = *(_QWORD *)(v7 + 48);
                                      v51 = v8;
                                      j_j___libc_free_0(v7);
                                      v6 = v51;
                                    }
                                    while ( v51 );
                                  }
                                  v52 = (_QWORD *)v5[2];
                                  sub_31541A0(v5[28]);
                                  v12 = v5[8];
                                  if ( (_QWORD *)v12 != v5 + 10 )
                                    _libc_free(v12);
                                  v13 = (_QWORD *)v5[6];
                                  if ( v13 )
                                    *v13 = v5[5];
                                  v14 = v5[5];
                                  if ( v14 )
                                    *(_QWORD *)(v14 + 8) = v5[6];
                                  j_j___libc_free_0((unsigned __int64)v5);
                                  v5 = v52;
                                }
                                while ( v52 );
                              }
                              v15 = (_QWORD *)v4[2];
                              sub_31541A0(v4[28]);
                              v16 = v4[8];
                              if ( (_QWORD *)v16 != v4 + 10 )
                                _libc_free(v16);
                              v17 = (_QWORD *)v4[6];
                              if ( v17 )
                                *v17 = v4[5];
                              v18 = v4[5];
                              if ( v18 )
                                *(_QWORD *)(v18 + 8) = v4[6];
                              v19 = (unsigned __int64)v4;
                              v4 = v15;
                              j_j___libc_free_0(v19);
                            }
                            while ( v15 );
                          }
                          v20 = (_QWORD *)v3[2];
                          sub_31541A0(v3[28]);
                          v21 = v3[8];
                          if ( (_QWORD *)v21 != v3 + 10 )
                            _libc_free(v21);
                          v22 = (_QWORD *)v3[6];
                          if ( v22 )
                            *v22 = v3[5];
                          v23 = v3[5];
                          if ( v23 )
                            *(_QWORD *)(v23 + 8) = v3[6];
                          j_j___libc_free_0((unsigned __int64)v3);
                          if ( !v20 )
                            break;
                          v3 = v20;
                        }
                      }
                      v24 = (_QWORD *)v2[2];
                      sub_31541A0(v2[28]);
                      v25 = v2[8];
                      if ( (_QWORD *)v25 != v2 + 10 )
                        _libc_free(v25);
                      v26 = (_QWORD *)v2[6];
                      if ( v26 )
                        *v26 = v2[5];
                      v27 = v2[5];
                      if ( v27 )
                        *(_QWORD *)(v27 + 8) = v2[6];
                      j_j___libc_free_0((unsigned __int64)v2);
                      if ( !v24 )
                        break;
                      v2 = v24;
                    }
                  }
                  v28 = (_QWORD *)v1[2];
                  sub_31541A0(v1[28]);
                  v29 = v1[8];
                  if ( (_QWORD *)v29 != v1 + 10 )
                    _libc_free(v29);
                  v30 = (_QWORD *)v1[6];
                  if ( v30 )
                    *v30 = v1[5];
                  v31 = v1[5];
                  if ( v31 )
                    *(_QWORD *)(v31 + 8) = v1[6];
                  j_j___libc_free_0((unsigned __int64)v1);
                  if ( !v28 )
                    break;
                  v1 = v28;
                }
              }
              v32 = (_QWORD *)v47[2];
              sub_31541A0(v47[28]);
              v33 = v47[8];
              if ( (_QWORD *)v33 != v47 + 10 )
                _libc_free(v33);
              v34 = (_QWORD *)v47[6];
              if ( v34 )
                *v34 = v47[5];
              v35 = v47[5];
              if ( v35 )
                *(_QWORD *)(v35 + 8) = v47[6];
              j_j___libc_free_0((unsigned __int64)v47);
              if ( !v32 )
                break;
              v47 = v32;
            }
          }
          v36 = (_QWORD *)v48[2];
          sub_31541A0(v48[28]);
          v37 = v48[8];
          if ( (_QWORD *)v37 != v48 + 10 )
            _libc_free(v37);
          v38 = (_QWORD *)v48[6];
          if ( v38 )
            *v38 = v48[5];
          v39 = v48[5];
          if ( v39 )
            *(_QWORD *)(v39 + 8) = v48[6];
          j_j___libc_free_0((unsigned __int64)v48);
          if ( !v36 )
            break;
          v48 = v36;
        }
      }
      v40 = (_QWORD *)v46[2];
      sub_31541A0(v46[28]);
      v41 = v46[8];
      if ( (_QWORD *)v41 != v46 + 10 )
        _libc_free(v41);
      v42 = (_QWORD *)v46[6];
      if ( v42 )
        *v42 = v46[5];
      v43 = v46[5];
      if ( v43 )
        *(_QWORD *)(v43 + 8) = v46[6];
      j_j___libc_free_0((unsigned __int64)v46);
      if ( !v40 )
        break;
      v46 = v40;
    }
  }
}
