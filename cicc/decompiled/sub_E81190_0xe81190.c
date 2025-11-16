// Function: sub_E81190
// Address: 0xe81190
//
char __fastcall sub_E81190(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 *a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v6; // rax
  _QWORD *v8; // r12
  __int64 *v9; // r10
  __int64 *v10; // r11
  __int64 v11; // r13
  __int64 *v13; // rbx
  __int64 v14; // r10
  __int64 *v15; // r11
  _QWORD *v16; // r9
  _QWORD *v17; // r8
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // edi
  __int64 v25; // r8
  int v26; // edi
  unsigned int v27; // r11d
  __int64 *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 *v31; // rdx
  __int64 v32; // r14
  unsigned __int64 v33; // r11
  __int64 v34; // r15
  _QWORD *v35; // rax
  __int64 v36; // r14
  _QWORD *v37; // rbx
  __int64 v39; // r13
  _QWORD *v40; // r8
  _QWORD *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  char v45; // al
  int v46; // edx
  int v47; // ecx
  int v48; // esi
  int v49; // ecx
  unsigned __int64 v51; // [rsp-90h] [rbp-90h]
  _QWORD *v52; // [rsp-88h] [rbp-88h]
  _QWORD *v53; // [rsp-88h] [rbp-88h]
  _QWORD *v54; // [rsp-88h] [rbp-88h]
  bool v55; // [rsp-72h] [rbp-72h]
  bool v56; // [rsp-71h] [rbp-71h]
  __int64 v57; // [rsp-70h] [rbp-70h]
  _QWORD *v58; // [rsp-70h] [rbp-70h]
  _QWORD *v59; // [rsp-68h] [rbp-68h]
  __int64 *v60; // [rsp-68h] [rbp-68h]
  __int64 *v61; // [rsp-68h] [rbp-68h]
  _QWORD *v62; // [rsp-60h] [rbp-60h]
  __int64 *v63; // [rsp-60h] [rbp-60h]
  __int64 v64; // [rsp-60h] [rbp-60h]
  _QWORD *v65; // [rsp-60h] [rbp-60h]
  __int64 *v66; // [rsp-60h] [rbp-60h]
  _QWORD *v67; // [rsp-60h] [rbp-60h]
  __int64 *v68; // [rsp-58h] [rbp-58h]
  _QWORD *v69; // [rsp-58h] [rbp-58h]
  char v70; // [rsp-58h] [rbp-58h]
  __int64 v71; // [rsp-58h] [rbp-58h]
  _QWORD *v72; // [rsp-58h] [rbp-58h]
  _QWORD *v73; // [rsp-58h] [rbp-58h]
  __int64 v74; // [rsp-50h] [rbp-50h]
  __int64 *v75; // [rsp-50h] [rbp-50h]
  bool v76; // [rsp-50h] [rbp-50h]
  _QWORD *v77; // [rsp-50h] [rbp-50h]
  __int64 v78; // [rsp-50h] [rbp-50h]
  __int64 *v79; // [rsp-50h] [rbp-50h]
  __int64 v80; // [rsp-50h] [rbp-50h]
  int v81; // [rsp-44h] [rbp-44h] BYREF
  __int64 v82; // [rsp-40h] [rbp-40h] BYREF

  v6 = *a4;
  if ( *a4 )
  {
    v8 = a5;
    if ( *a5 )
    {
      v9 = *(__int64 **)(v6 + 16);
      v10 = *(__int64 **)(*a5 + 16LL);
      v11 = a1;
      v13 = a4;
      if ( *v9 )
        goto LABEL_96;
      v69 = a6;
      v75 = *(__int64 **)(*a5 + 16LL);
      LOBYTE(v6) = *((_BYTE *)v9 + 9) & 0x70;
      if ( (_BYTE)v6 == 32 && *((char *)v9 + 8) >= 0 )
      {
        *((_BYTE *)v9 + 8) |= 8u;
        v63 = v9;
        v6 = (__int64)sub_E807D0(v9[3]);
        v9 = v63;
        v10 = v75;
        a6 = v69;
        *v63 = v6;
        if ( v6 )
        {
LABEL_96:
          if ( *v10 )
            goto LABEL_95;
          v72 = a6;
          v79 = v9;
          LOBYTE(v6) = *((_BYTE *)v10 + 9) & 0x70;
          if ( (_BYTE)v6 == 32 && *((char *)v10 + 8) >= 0 )
          {
            *((_BYTE *)v10 + 8) |= 8u;
            v66 = v10;
            v6 = (__int64)sub_E807D0(v10[3]);
            v10 = v66;
            v9 = v79;
            a6 = v72;
            *v66 = v6;
            if ( v6 )
            {
LABEL_95:
              v62 = a6;
              v68 = v10;
              v74 = (__int64)v9;
              LOBYTE(v6) = sub_E8ECE0(*(_QWORD *)(a1 + 24), a1, *v13, *v8, a3);
              if ( (_BYTE)v6 )
              {
                v14 = v74;
                v15 = v68;
                v16 = v62;
                v17 = *(_QWORD **)v74;
                if ( !*(_QWORD *)v74 && (*(_BYTE *)(v74 + 9) & 0x70) == 0x20 && *(char *)(v74 + 8) >= 0 )
                {
                  *(_BYTE *)(v74 + 8) |= 8u;
                  v42 = sub_E807D0(*(_QWORD *)(v74 + 24));
                  v14 = v74;
                  v16 = v62;
                  v15 = v68;
                  v17 = v42;
                  *(_QWORD *)v74 = v42;
                }
                v18 = (_QWORD *)*v15;
                if ( !*v15 )
                {
                  v67 = v16;
                  v73 = v17;
                  v80 = v14;
                  if ( (*((_BYTE *)v15 + 9) & 0x70) != 0x20 || *((char *)v15 + 8) < 0 )
                    BUG();
                  *((_BYTE *)v15 + 8) |= 8u;
                  v61 = v15;
                  v41 = sub_E807D0(v15[3]);
                  v15 = v61;
                  v16 = v67;
                  v17 = v73;
                  v14 = v80;
                  v18 = v41;
                  *v61 = (__int64)v41;
                }
                v19 = v17[1];
                v64 = v18[1];
                v76 = v19 != v64;
                LOBYTE(v6) = a2 == 0;
                if ( a2 != 0 || v19 == v64 )
                {
                  v70 = *(_BYTE *)(a1 + 32);
                  if ( v70 && (a3 || (*(_BYTE *)(v19 + 48) & 2) == 0 || *(_DWORD *)(*(_QWORD *)(a1 + 8) + 12LL) == 1320) )
                  {
                    if ( v18 != v17 || (*(_BYTE *)(v14 + 9) & 0x70) == 0x20 || (*((_BYTE *)v15 + 9) & 0x70) == 0x20 )
                    {
                      v59 = v16;
                      v71 = v14;
                      v57 = v17[1];
                      v20 = sub_E5C4C0(a1, *(_QWORD *)(*v13 + 16));
                      v21 = sub_E5C4C0(a1, *(_QWORD *)(*v8 + 16LL));
                      v16 = v59;
                      v14 = v71;
                      v22 = *v59 + v20 - v21;
                      *v59 = v22;
                      v23 = v22;
                      if ( a2 && v76 )
                      {
                        v24 = *(_DWORD *)(a2 + 24);
                        v25 = *(_QWORD *)(a2 + 8);
                        if ( v24 )
                        {
                          v26 = v24 - 1;
                          v27 = v26 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
                          v28 = (__int64 *)(v25 + 16LL * v27);
                          v29 = *v28;
                          if ( v57 == *v28 )
                          {
LABEL_23:
                            v22 += v28[1];
                          }
                          else
                          {
                            v48 = 1;
                            while ( v29 != -4096 )
                            {
                              v49 = v48 + 1;
                              v27 = v26 & (v48 + v27);
                              v28 = (__int64 *)(v25 + 16LL * v27);
                              v29 = *v28;
                              if ( v57 == *v28 )
                                goto LABEL_23;
                              v48 = v49;
                            }
                          }
                          v30 = v26 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                          v31 = (__int64 *)(v25 + 16 * v30);
                          v32 = *v31;
                          if ( v64 == *v31 )
                          {
LABEL_25:
                            v22 -= v31[1];
                          }
                          else
                          {
                            v46 = 1;
                            while ( v32 != -4096 )
                            {
                              v47 = v46 + 1;
                              LODWORD(v30) = v26 & (v30 + v46);
                              v31 = (__int64 *)(v25 + 16LL * (unsigned int)v30);
                              v32 = *v31;
                              if ( v64 == *v31 )
                                goto LABEL_25;
                              v46 = v47;
                            }
                          }
                          v23 = v22;
                        }
                        *v59 = v23;
                      }
                    }
                    else
                    {
                      *v16 += *(_QWORD *)(v14 + 24) - v15[3];
                    }
                    goto LABEL_28;
                  }
                  LOBYTE(v6) = *(_BYTE *)(v14 + 9) & 0x70;
                  if ( (_BYTE)v6 != 32 )
                  {
                    LOBYTE(v6) = *((_BYTE *)v15 + 9) & 0x70;
                    if ( (_BYTE)v6 != 32 )
                    {
                      v33 = v15[3];
                      v78 = *(_QWORD *)(v14 + 24);
                      if ( v18 == v17 )
                        v56 = v33 > *(_QWORD *)(v14 + 24);
                      else
                        v56 = *((_DWORD *)v17 + 6) < *((_DWORD *)v18 + 6);
                      v34 = v78 - v33;
                      if ( v56 )
                      {
                        v78 = v33;
                        v33 = *(_QWORD *)(v14 + 24);
                        v35 = v17;
                        v34 = -v34;
                        v17 = v18;
                        v18 = v35;
                      }
                      v55 = 0;
                      v36 = (__int64)v18;
                      v51 = v33;
                      v65 = v18;
                      v60 = v13;
                      v37 = v17;
                      v58 = v8;
                      v39 = v14;
                      do
                      {
                        LOBYTE(v6) = *(_BYTE *)(v36 + 28);
                        if ( (_BYTE)v6 == 1 )
                        {
                          LOBYTE(v6) = (*(_BYTE *)(v36 + 29) & 4) != 0;
                          if ( (*(_BYTE *)(v36 + 29) & 4) != 0 )
                          {
                            if ( v65 != (_QWORD *)v36 || *(_QWORD *)(v36 + 48) != v51 )
                            {
                              v40 = v37;
                              v14 = v39;
                              v13 = v60;
                              v11 = a1;
                              v8 = v58;
                              if ( v40 != (_QWORD *)v36
                                || *(_QWORD *)(v36 + 48) == v78
                                || (*(_BYTE *)(v36 + 29) & 4) != 0 && v55 )
                              {
                                return v6;
                              }
LABEL_44:
                              if ( v56 )
                                v34 = -v34;
                              *v16 += v34;
LABEL_28:
                              v77 = v16;
                              LOBYTE(v6) = sub_E5BBB0(v11, v14);
                              if ( (_BYTE)v6 )
                                *v77 |= 1uLL;
                              *v8 = 0;
                              *v13 = 0;
                              return v6;
                            }
                            if ( v37 == v65 )
                            {
LABEL_82:
                              v14 = v39;
                              v13 = v60;
                              v11 = a1;
                              v8 = v58;
                              goto LABEL_44;
                            }
                            v55 = (*(_BYTE *)(v36 + 29) & 4) != 0;
                          }
                          else if ( v37 == (_QWORD *)v36 )
                          {
                            goto LABEL_82;
                          }
                          v34 += *(_QWORD *)(v36 + 48);
                        }
                        else
                        {
                          if ( v37 == (_QWORD *)v36 )
                            goto LABEL_82;
                          if ( !(_BYTE)v6 )
                          {
                            if ( !v70 || (*(_BYTE *)(v36 + 31) & 1) == 0 )
                              return v6;
                            v43 = *(_QWORD *)(a1 + 8);
                            v44 = *(__int64 (**)())(*(_QWORD *)v43 + 80LL);
                            if ( v44 == sub_E5B800
                              || (v53 = v16,
                                  v45 = ((__int64 (__fastcall *)(__int64, __int64, int *))v44)(v43, v36, &v81),
                                  v16 = v53,
                                  !v45) )
                            {
                              v52 = v16;
                              v6 = sub_E5BD20((__int64 *)a1, v36);
                              v16 = v52;
                              v34 += v6;
                              goto LABEL_64;
                            }
                            LOBYTE(v6) = *(_BYTE *)(v36 + 28);
                          }
                          if ( (_BYTE)v6 != 2 )
                            return v6;
                          v54 = v16;
                          LOBYTE(v6) = sub_E81180(*(_QWORD *)(v36 + 40), &v82);
                          if ( !(_BYTE)v6 )
                            return v6;
                          v6 = v82 * *(unsigned __int8 *)(v36 + 30);
                          v16 = v54;
                          v34 += v6;
                        }
LABEL_64:
                        v36 = *(_QWORD *)v36;
                      }
                      while ( v36 );
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
  return v6;
}
