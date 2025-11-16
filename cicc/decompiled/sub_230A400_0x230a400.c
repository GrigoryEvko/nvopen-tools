// Function: sub_230A400
// Address: 0x230a400
//
_QWORD *__fastcall sub_230A400(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  void **v4; // r12
  _QWORD *v5; // rax
  void **v6; // r15
  _QWORD *v7; // rax
  void **v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // r13
  _QWORD *v14; // r14
  _QWORD *v15; // rcx
  _QWORD *v16; // r8
  _QWORD *v17; // rbx
  __int64 v18; // r14
  _QWORD *v19; // r13
  _QWORD *v20; // rax
  _QWORD *v21; // r10
  _QWORD *v22; // r15
  _QWORD *v23; // rax
  _QWORD *v24; // rcx
  _QWORD *v25; // r9
  _QWORD *v26; // rax
  _QWORD *v27; // r11
  _QWORD *v28; // r12
  _QWORD *v29; // rax
  _QWORD **v30; // r8
  _QWORD **v31; // rbx
  _QWORD *v32; // rax
  _QWORD *v33; // r15
  __int64 v34; // rdx
  _QWORD *v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // r15
  _QWORD *v41; // rax
  __int64 v42; // r14
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  _QWORD *v45; // rdi
  _QWORD *v46; // rax
  _QWORD *v47; // r13
  _QWORD *v48; // rcx
  _QWORD *v49; // r8
  _QWORD *v50; // rbx
  __int64 v51; // r13
  _QWORD *v52; // r14
  _QWORD *v53; // rax
  _QWORD *v54; // r10
  _QWORD *v55; // r15
  _QWORD *v56; // rax
  _QWORD *v57; // rcx
  _QWORD *v58; // r9
  _QWORD *v59; // rax
  _QWORD *v60; // r11
  _QWORD *v61; // r12
  _QWORD *v62; // rax
  _QWORD **v63; // r8
  _QWORD **v64; // rbx
  _QWORD *v66; // [rsp+8h] [rbp-138h]
  void **v67; // [rsp+10h] [rbp-130h]
  _QWORD *v68; // [rsp+18h] [rbp-128h]
  _QWORD *v69; // [rsp+18h] [rbp-128h]
  _QWORD *v71; // [rsp+30h] [rbp-110h]
  _QWORD *v72; // [rsp+30h] [rbp-110h]
  _QWORD *v73; // [rsp+38h] [rbp-108h]
  _QWORD *v74; // [rsp+38h] [rbp-108h]
  _QWORD *v75; // [rsp+40h] [rbp-100h]
  _QWORD *v76; // [rsp+40h] [rbp-100h]
  _QWORD *v77; // [rsp+48h] [rbp-F8h]
  _QWORD *v78; // [rsp+48h] [rbp-F8h]
  _QWORD *v79; // [rsp+58h] [rbp-E8h]
  _QWORD *v80; // [rsp+58h] [rbp-E8h]
  _QWORD *v81; // [rsp+60h] [rbp-E0h]
  _QWORD *v82; // [rsp+60h] [rbp-E0h]
  _QWORD *v83; // [rsp+68h] [rbp-D8h]
  _QWORD *v84; // [rsp+68h] [rbp-D8h]
  __int64 v85; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+78h] [rbp-C8h]
  __int64 v87; // [rsp+80h] [rbp-C0h]
  __int64 v88; // [rsp+88h] [rbp-B8h]
  _QWORD *v89; // [rsp+90h] [rbp-B0h]
  __int64 v90; // [rsp+98h] [rbp-A8h]
  __int64 v91; // [rsp+A0h] [rbp-A0h]
  __int64 v92; // [rsp+A8h] [rbp-98h]
  int v93; // [rsp+B0h] [rbp-90h]
  void *v94; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v95; // [rsp+C8h] [rbp-78h]
  __int64 v96; // [rsp+D0h] [rbp-70h]
  __int64 v97; // [rsp+D8h] [rbp-68h]
  _QWORD *v98; // [rsp+E0h] [rbp-60h]
  __int64 v99; // [rsp+E8h] [rbp-58h]
  __int64 v100; // [rsp+F0h] [rbp-50h]
  __int64 v101; // [rsp+F8h] [rbp-48h]
  int v102; // [rsp+100h] [rbp-40h]

  v4 = &v94;
  sub_22E2690(&v85, a2 + 8, a3, a4);
  v99 = 1;
  v90 += 2;
  v100 = v91;
  v95 = v86;
  v101 = v92;
  v96 = v87;
  v97 = v88;
  v5 = v89;
  v102 = v93;
  v98 = v89;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v94 = &unk_4A0A0C0;
  if ( v98 )
  {
    v5[2] = &v94;
    v75 = (_QWORD *)v5[6];
    if ( (_QWORD *)v5[5] != v75 )
    {
      v83 = (_QWORD *)v5[5];
      v6 = &v94;
      do
      {
        v7 = (_QWORD *)*v83;
        if ( *v83 )
        {
          v7[2] = v6;
          v73 = (_QWORD *)v7[6];
          if ( (_QWORD *)v7[5] != v73 )
          {
            v79 = (_QWORD *)v7[5];
            v8 = v6;
            do
            {
              v9 = (_QWORD *)*v79;
              if ( *v79 )
              {
                v9[2] = v8;
                v71 = (_QWORD *)v9[6];
                if ( (_QWORD *)v9[5] != v71 )
                {
                  v81 = (_QWORD *)v9[5];
                  do
                  {
                    v10 = (_QWORD *)*v81;
                    if ( *v81 )
                    {
                      v11 = (_QWORD *)v10[6];
                      v12 = (_QWORD *)v10[5];
                      v10[2] = v8;
                      if ( v12 != v11 )
                      {
                        v13 = (__int64)v8;
                        v14 = v12;
                        do
                        {
                          v15 = (_QWORD *)*v14;
                          if ( *v14 )
                          {
                            v16 = (_QWORD *)v15[5];
                            v17 = (_QWORD *)v15[6];
                            v15[2] = v13;
                            if ( v16 != v17 )
                            {
                              v77 = v14;
                              v18 = v13;
                              v19 = v16;
                              do
                              {
                                v20 = (_QWORD *)*v19;
                                if ( *v19 )
                                {
                                  v21 = (_QWORD *)v20[5];
                                  v22 = (_QWORD *)v20[6];
                                  v20[2] = v18;
                                  if ( v21 != v22 )
                                  {
                                    v68 = v17;
                                    do
                                    {
                                      v23 = (_QWORD *)*v21;
                                      if ( *v21 )
                                      {
                                        v24 = (_QWORD *)v23[5];
                                        v25 = (_QWORD *)v23[6];
                                        for ( v23[2] = v18; v25 != v24; ++v24 )
                                        {
                                          v26 = (_QWORD *)*v24;
                                          if ( *v24 )
                                          {
                                            v27 = (_QWORD *)v26[5];
                                            v28 = (_QWORD *)v26[6];
                                            for ( v26[2] = v18; v28 != v27; ++v27 )
                                            {
                                              v29 = (_QWORD *)*v27;
                                              if ( *v27 )
                                              {
                                                v30 = (_QWORD **)v29[5];
                                                v31 = (_QWORD **)v29[6];
                                                v29[2] = v18;
                                                while ( v31 != v30 )
                                                  sub_2306AF0(v18, *v30);
                                              }
                                            }
                                          }
                                        }
                                      }
                                      ++v21;
                                    }
                                    while ( v22 != v21 );
                                    v17 = v68;
                                  }
                                }
                                ++v19;
                              }
                              while ( v17 != v19 );
                              v13 = v18;
                              v14 = v77;
                            }
                          }
                          ++v14;
                        }
                        while ( v11 != v14 );
                        v8 = (void **)v13;
                      }
                    }
                    ++v81;
                  }
                  while ( v71 != v81 );
                }
              }
              ++v79;
            }
            while ( v73 != v79 );
            v6 = v8;
          }
        }
        ++v83;
      }
      while ( v75 != v83 );
      v4 = v6;
    }
  }
  v32 = (_QWORD *)sub_22077B0(0x50u);
  v33 = v32;
  if ( v32 )
  {
    v34 = v100;
    v32[6] = 1;
    v35 = v32 + 1;
    v99 += 2;
    v32[7] = v34;
    v36 = v101;
    *v32 = &unk_4A0AE58;
    v37 = v95;
    v33[8] = v36;
    LODWORD(v36) = v102;
    v33[2] = v37;
    v38 = v96;
    *((_DWORD *)v33 + 18) = v36;
    v33[3] = v38;
    v100 = 0;
    v33[4] = v97;
    v39 = v98;
    v101 = 0;
    v33[5] = v98;
    v102 = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    v33[1] = &unk_4A0A0C0;
    if ( v39 )
    {
      v39[2] = v35;
      v76 = (_QWORD *)v39[6];
      if ( (_QWORD *)v39[5] != v76 )
      {
        v84 = (_QWORD *)v39[5];
        v67 = v4;
        v69 = v33;
        v40 = v33 + 1;
        do
        {
          v41 = (_QWORD *)*v84;
          if ( *v84 )
          {
            v41[2] = v40;
            v74 = (_QWORD *)v41[6];
            if ( (_QWORD *)v41[5] != v74 )
            {
              v80 = (_QWORD *)v41[5];
              v42 = (__int64)v40;
              do
              {
                v43 = (_QWORD *)*v80;
                if ( *v80 )
                {
                  v43[2] = v42;
                  v72 = (_QWORD *)v43[6];
                  if ( (_QWORD *)v43[5] != v72 )
                  {
                    v82 = (_QWORD *)v43[5];
                    do
                    {
                      v44 = (_QWORD *)*v82;
                      if ( *v82 )
                      {
                        v45 = (_QWORD *)v44[6];
                        v46 = (_QWORD *)v44[5];
                        v44[2] = v42;
                        if ( v46 != v45 )
                        {
                          v47 = v46;
                          do
                          {
                            v48 = (_QWORD *)*v47;
                            if ( *v47 )
                            {
                              v49 = (_QWORD *)v48[5];
                              v50 = (_QWORD *)v48[6];
                              v48[2] = v42;
                              if ( v49 != v50 )
                              {
                                v78 = v47;
                                v51 = v42;
                                v52 = v49;
                                do
                                {
                                  v53 = (_QWORD *)*v52;
                                  if ( *v52 )
                                  {
                                    v54 = (_QWORD *)v53[5];
                                    v55 = (_QWORD *)v53[6];
                                    v53[2] = v51;
                                    if ( v54 != v55 )
                                    {
                                      v66 = v50;
                                      do
                                      {
                                        v56 = (_QWORD *)*v54;
                                        if ( *v54 )
                                        {
                                          v57 = (_QWORD *)v56[5];
                                          v58 = (_QWORD *)v56[6];
                                          for ( v56[2] = v51; v58 != v57; ++v57 )
                                          {
                                            v59 = (_QWORD *)*v57;
                                            if ( *v57 )
                                            {
                                              v60 = (_QWORD *)v59[5];
                                              v61 = (_QWORD *)v59[6];
                                              for ( v59[2] = v51; v61 != v60; ++v60 )
                                              {
                                                v62 = (_QWORD *)*v60;
                                                if ( *v60 )
                                                {
                                                  v63 = (_QWORD **)v62[5];
                                                  v64 = (_QWORD **)v62[6];
                                                  v62[2] = v51;
                                                  while ( v64 != v63 )
                                                    sub_2306AF0(v51, *v63);
                                                }
                                              }
                                            }
                                          }
                                        }
                                        ++v54;
                                      }
                                      while ( v55 != v54 );
                                      v50 = v66;
                                    }
                                  }
                                  ++v52;
                                }
                                while ( v50 != v52 );
                                v42 = v51;
                                v47 = v78;
                              }
                            }
                            ++v47;
                          }
                          while ( v45 != v47 );
                        }
                      }
                      ++v82;
                    }
                    while ( v72 != v82 );
                  }
                }
                ++v80;
              }
              while ( v74 != v80 );
              v40 = (_QWORD *)v42;
            }
          }
          ++v84;
        }
        while ( v76 != v84 );
        v33 = v69;
        v4 = v67;
      }
    }
  }
  sub_22DC7A0((__int64)v4);
  *a1 = v33;
  sub_22DC7A0((__int64)&v85);
  return a1;
}
