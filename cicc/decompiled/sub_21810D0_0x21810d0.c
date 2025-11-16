// Function: sub_21810D0
// Address: 0x21810d0
//
void __fastcall sub_21810D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v7; // eax
  __int64 v8; // r8
  int v9; // r9d
  size_t v10; // rdi
  int v11; // r13d
  __int64 v12; // rax
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdi
  __int64 *v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 *v21; // r8
  __int64 v22; // rcx
  __int64 *v23; // r15
  __int64 *v24; // rbx
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // r9
  void *v29; // rax
  int v30; // r11d
  __int64 *v31; // r10
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r10
  __int64 v36; // r9
  void *v37; // rax
  __int64 v38; // [rsp-88h] [rbp-88h]
  __int64 v39; // [rsp-80h] [rbp-80h]
  __int64 v40; // [rsp-80h] [rbp-80h]
  __int64 v41; // [rsp-80h] [rbp-80h]
  unsigned int v42; // [rsp-78h] [rbp-78h]
  unsigned int v43; // [rsp-78h] [rbp-78h]
  __int64 v44; // [rsp-78h] [rbp-78h]
  __int64 v45; // [rsp-78h] [rbp-78h]
  __int64 v46; // [rsp-70h] [rbp-70h]
  __int64 v47; // [rsp-70h] [rbp-70h]
  __int64 v48; // [rsp-70h] [rbp-70h]
  __int64 v49; // [rsp-70h] [rbp-70h]
  __int64 v50; // [rsp-68h] [rbp-68h] BYREF
  __int64 *v51; // [rsp-60h] [rbp-60h] BYREF
  __int64 v52; // [rsp-58h] [rbp-58h] BYREF
  __int64 *v53; // [rsp-50h] [rbp-50h]
  __int64 v54; // [rsp-48h] [rbp-48h]
  __int64 v55; // [rsp-40h] [rbp-40h]

  if ( a4 != *(_QWORD *)(a3 + 24) )
  {
    v7 = sub_217DB60(a3);
    v10 = *(_QWORD *)(a1 + 248);
    v11 = v7;
    v12 = v7 < 0
        ? *(_QWORD *)(*(_QWORD *)(v10 + 24) + 16LL * (v7 & 0x7FFFFFFF) + 8)
        : *(_QWORD *)(*(_QWORD *)(v10 + 272) + 8LL * (unsigned int)v7);
    if ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            goto LABEL_6;
        }
      }
      else
      {
LABEL_6:
        v13 = sub_1E6B9A0(
                v10,
                *(_QWORD *)(*(_QWORD *)(v10 + 24) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                (unsigned __int8 *)byte_3F871B3,
                0,
                v8,
                v9);
        v14 = sub_1DD5D10(a4);
        (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256)
                                                                                          + 152LL))(
          *(_QWORD *)(a1 + 256),
          a4,
          v14,
          v13,
          0,
          a3,
          *(_QWORD *)(a1 + 232));
        v52 = 0;
        v53 = 0;
        v15 = *(_QWORD *)(a1 + 248);
        v54 = 0;
        v55 = 0;
        if ( v11 < 0 )
          v16 = *(__int64 **)(*(_QWORD *)(v15 + 24) + 16LL * (v11 & 0x7FFFFFFF) + 8);
        else
          v16 = *(__int64 **)(*(_QWORD *)(v15 + 272) + 8LL * (unsigned int)v11);
        while ( v16 )
        {
          if ( (*((_BYTE *)v16 + 3) & 0x10) == 0 && (*((_BYTE *)v16 + 4) & 8) == 0 )
          {
            v17 = v16;
            v18 = 0;
            v16 = 0;
LABEL_21:
            v19 = v17[2];
            v50 = v19;
            if ( (_DWORD)v18 )
            {
              v20 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
              v21 = &v16[v20];
              v22 = *v21;
              if ( v19 == *v21 )
              {
LABEL_23:
                while ( 1 )
                {
                  v17 = (__int64 *)v17[4];
                  if ( !v17 )
                    break;
                  while ( (*((_BYTE *)v17 + 3) & 0x10) == 0 )
                  {
                    if ( (*((_BYTE *)v17 + 4) & 8) == 0 )
                      goto LABEL_21;
                    v17 = (__int64 *)v17[4];
                    if ( !v17 )
                      goto LABEL_27;
                  }
                }
LABEL_27:
                v23 = &v16[v18];
                if ( (_DWORD)v54 && v23 != v16 )
                {
                  v24 = v16;
                  while ( *v24 == -16 || *v24 == -8 )
                  {
                    if ( ++v24 == v23 )
                      goto LABEL_12;
                  }
                  if ( v23 != v24 )
                  {
LABEL_35:
                    v25 = *v24;
                    if ( **(_WORD **)(*v24 + 16) == 45 || !**(_WORD **)(*v24 + 16) )
                    {
                      if ( *(_DWORD *)(v25 + 40) != 1 )
                      {
                        v26 = 1;
                        do
                        {
                          v27 = *(_QWORD *)(v25 + 32);
                          v28 = v27 + 40LL * v26;
                          if ( !*(_BYTE *)v28
                            && v11 == *(_DWORD *)(v28 + 8)
                            && a4 == *(_QWORD *)(v27 + 40LL * (v26 + 1) + 24) )
                          {
                            if ( byte_4FD2E80 )
                            {
                              v39 = v27 + 40LL * v26;
                              v43 = v26;
                              v47 = v25;
                              v29 = sub_16E8CB0();
                              sub_1263B40((__int64)v29, "\treplaced\n");
                              v28 = v39;
                              v26 = v43;
                              v25 = v47;
                            }
                            v42 = v26;
                            v46 = v25;
                            sub_1E310D0(v28, v13);
                            v26 = v42;
                            v25 = v46;
                          }
                          v26 += 2;
                        }
                        while ( v26 != *(_DWORD *)(v25 + 40) );
                      }
                    }
                    else if ( a4 == *(_QWORD *)(v25 + 24) )
                    {
                      v33 = *(unsigned int *)(v25 + 40);
                      if ( (_DWORD)v33 )
                      {
                        v34 = 0;
                        v35 = 40 * v33;
                        do
                        {
                          v36 = v34 + *(_QWORD *)(v25 + 32);
                          if ( !*(_BYTE *)v36 && (*(_BYTE *)(v36 + 3) & 0x10) == 0 && v11 == *(_DWORD *)(v36 + 8) )
                          {
                            if ( byte_4FD2E80 )
                            {
                              v38 = v35;
                              v41 = v34;
                              v45 = v34 + *(_QWORD *)(v25 + 32);
                              v49 = v25;
                              v37 = sub_16E8CB0();
                              sub_1263B40((__int64)v37, "\treplaced\n");
                              v35 = v38;
                              v34 = v41;
                              v36 = v45;
                              v25 = v49;
                            }
                            v40 = v35;
                            v44 = v34;
                            v48 = v25;
                            sub_1E310D0(v36, v13);
                            v35 = v40;
                            v34 = v44;
                            v25 = v48;
                          }
                          v34 += 40;
                        }
                        while ( v35 != v34 );
                      }
                    }
                    while ( ++v24 != v23 )
                    {
                      if ( *v24 != -16 && *v24 != -8 )
                      {
                        if ( v24 != v23 )
                          goto LABEL_35;
                        break;
                      }
                    }
                    v16 = v53;
                  }
                }
                break;
              }
              v30 = 1;
              v31 = 0;
              while ( v22 != -8 )
              {
                if ( v31 || v22 != -16 )
                  v21 = v31;
                v20 = (v18 - 1) & (v30 + v20);
                v22 = v16[v20];
                if ( v19 == v22 )
                  goto LABEL_23;
                ++v30;
                v31 = v21;
                v21 = &v16[v20];
              }
              if ( !v31 )
                v31 = v21;
              ++v52;
              v32 = v54 + 1;
              if ( 4 * ((int)v54 + 1) < (unsigned int)(3 * v18) )
              {
                if ( (int)v18 - (v32 + HIDWORD(v54)) > (unsigned int)v18 >> 3 )
                {
LABEL_58:
                  LODWORD(v54) = v32;
                  if ( *v31 != -8 )
                    --HIDWORD(v54);
                  *v31 = v19;
                  v16 = v53;
                  v18 = (unsigned int)v55;
                  goto LABEL_23;
                }
LABEL_63:
                sub_1E22DE0((__int64)&v52, v18);
                sub_1E1F3B0((__int64)&v52, &v50, &v51);
                v31 = v51;
                v19 = v50;
                v32 = v54 + 1;
                goto LABEL_58;
              }
            }
            else
            {
              ++v52;
            }
            LODWORD(v18) = 2 * v18;
            goto LABEL_63;
          }
          v16 = (__int64 *)v16[4];
        }
LABEL_12:
        j___libc_free_0(v16);
      }
    }
  }
}
