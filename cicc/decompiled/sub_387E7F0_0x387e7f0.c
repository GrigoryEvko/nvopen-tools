// Function: sub_387E7F0
// Address: 0x387e7f0
//
void __fastcall sub_387E7F0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  _QWORD *v15; // r14
  _QWORD *v16; // r15
  unsigned __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  double v21; // xmm4_8
  double v22; // xmm5_8
  unsigned __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // rdi
  _QWORD *v26; // rbx
  __int64 v27; // rdi
  _QWORD *v28; // rbx
  __int64 v29; // rdi
  _QWORD *v30; // rbx
  __int64 v31; // rdi
  _QWORD *v32; // rbx
  __int64 v33; // rdi
  _QWORD *v34; // rbx
  __int64 v35; // rdi
  _QWORD *v36; // rbx
  __int64 v37; // rdi
  _QWORD *v38; // rbx
  __int64 v39; // rdi
  _QWORD *v40; // rbx
  unsigned __int64 v41; // [rsp+8h] [rbp-58h]
  _QWORD *v42; // [rsp+10h] [rbp-50h]
  _QWORD *v43; // [rsp+18h] [rbp-48h]
  _QWORD *v44; // [rsp+20h] [rbp-40h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]

  v42 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v45 = (_QWORD *)v42[3];
      if ( v45 )
      {
        while ( 1 )
        {
          v44 = (_QWORD *)v45[3];
          if ( v44 )
          {
            while ( 1 )
            {
              v43 = (_QWORD *)v44[3];
              if ( v43 )
              {
                while ( 1 )
                {
                  v13 = (_QWORD *)v43[3];
                  if ( v13 )
                  {
                    while ( 1 )
                    {
                      v14 = (_QWORD *)v13[3];
                      if ( v14 )
                      {
                        while ( 1 )
                        {
                          v15 = (_QWORD *)v14[3];
                          if ( v15 )
                          {
                            while ( 1 )
                            {
                              v16 = (_QWORD *)v15[3];
                              if ( v16 )
                              {
                                while ( 1 )
                                {
                                  v17 = v16[3];
                                  while ( v17 )
                                  {
                                    sub_387E7F0(*(_QWORD *)(v17 + 24));
                                    v23 = v17;
                                    v17 = *(_QWORD *)(v17 + 16);
                                    v24 = *(_QWORD *)(v23 + 40);
                                    if ( v24 )
                                    {
                                      v41 = v23;
                                      sub_16307F0(v24, a2, v18, v19, v20, a6, a7, a8, a9, v21, v22, a12, a13);
                                      v23 = v41;
                                    }
                                    a2 = 56;
                                    j_j___libc_free_0(v23);
                                  }
                                  v25 = v16[5];
                                  v26 = (_QWORD *)v16[2];
                                  if ( v25 )
                                    sub_16307F0(v25, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                                  a2 = 56;
                                  j_j___libc_free_0((unsigned __int64)v16);
                                  if ( !v26 )
                                    break;
                                  v16 = v26;
                                }
                              }
                              v33 = v15[5];
                              v34 = (_QWORD *)v15[2];
                              if ( v33 )
                                sub_16307F0(v33, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                              a2 = 56;
                              j_j___libc_free_0((unsigned __int64)v15);
                              if ( !v34 )
                                break;
                              v15 = v34;
                            }
                          }
                          v29 = v14[5];
                          v30 = (_QWORD *)v14[2];
                          if ( v29 )
                            sub_16307F0(v29, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                          a2 = 56;
                          j_j___libc_free_0((unsigned __int64)v14);
                          if ( !v30 )
                            break;
                          v14 = v30;
                        }
                      }
                      v27 = v13[5];
                      v28 = (_QWORD *)v13[2];
                      if ( v27 )
                        sub_16307F0(v27, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                      a2 = 56;
                      j_j___libc_free_0((unsigned __int64)v13);
                      if ( !v28 )
                        break;
                      v13 = v28;
                    }
                  }
                  v31 = v43[5];
                  v32 = (_QWORD *)v43[2];
                  if ( v31 )
                    sub_16307F0(v31, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
                  a2 = 56;
                  j_j___libc_free_0((unsigned __int64)v43);
                  if ( !v32 )
                    break;
                  v43 = v32;
                }
              }
              v35 = v44[5];
              v36 = (_QWORD *)v44[2];
              if ( v35 )
                sub_16307F0(v35, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
              a2 = 56;
              j_j___libc_free_0((unsigned __int64)v44);
              if ( !v36 )
                break;
              v44 = v36;
            }
          }
          v37 = v45[5];
          v38 = (_QWORD *)v45[2];
          if ( v37 )
            sub_16307F0(v37, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
          a2 = 56;
          j_j___libc_free_0((unsigned __int64)v45);
          if ( !v38 )
            break;
          v45 = v38;
        }
      }
      v39 = v42[5];
      v40 = (_QWORD *)v42[2];
      if ( v39 )
        sub_16307F0(v39, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
      a2 = 56;
      j_j___libc_free_0((unsigned __int64)v42);
      if ( !v40 )
        break;
      v42 = v40;
    }
  }
}
