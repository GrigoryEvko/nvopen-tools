// Function: sub_6EB6C0
// Address: 0x6eb6c0
//
_BOOL8 __fastcall sub_6EB6C0(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        unsigned int a4,
        int a5,
        int a6,
        int a7,
        int a8,
        __int64 *a9)
{
  __int64 v10; // r15
  __int64 v11; // r14
  int v12; // edx
  __int64 v13; // r8
  _BOOL4 v14; // esi
  __int64 v15; // rax
  int v16; // eax
  int v17; // r8d
  _BOOL4 v18; // eax
  __int64 v19; // rax
  _BOOL8 v20; // r12
  __int64 v21; // r9
  _BOOL4 v23; // eax
  __int64 v24; // r9
  __int64 v25; // rdi
  __int64 v26; // rax
  char i; // dl
  char v28; // bl
  _BOOL4 v29; // eax
  int v30; // eax
  _BOOL4 v31; // eax
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 n; // rax
  __int64 v38; // r13
  __int64 ii; // rax
  char v40; // si
  __int64 v41; // rbx
  __int64 v42; // rax
  char v43; // dl
  char v44; // di
  __int64 v45; // rcx
  char v46; // al
  __int64 k; // r13
  __int64 m; // r12
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // eax
  int v53; // eax
  int v54; // eax
  unsigned int v55; // esi
  __int64 v56; // rax
  char j; // dl
  __int64 v58; // r12
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdi
  _BOOL4 v62; // eax
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // eax
  int v69; // eax
  __int64 v70; // rsi
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rax
  _BOOL4 v74; // eax
  unsigned int v75; // r8d
  int v79; // [rsp+10h] [rbp-80h]
  __int64 v80; // [rsp+18h] [rbp-78h]
  __int64 v81; // [rsp+20h] [rbp-70h]
  _BOOL4 v83; // [rsp+28h] [rbp-68h]
  __int64 v84; // [rsp+28h] [rbp-68h]
  __int64 v85; // [rsp+28h] [rbp-68h]
  __int64 v86; // [rsp+28h] [rbp-68h]
  __int64 v87; // [rsp+28h] [rbp-68h]
  __int64 v88; // [rsp+28h] [rbp-68h]
  __int64 v89; // [rsp+28h] [rbp-68h]
  int v90; // [rsp+30h] [rbp-60h]
  __int64 v91; // [rsp+30h] [rbp-60h]
  unsigned int v92; // [rsp+38h] [rbp-58h]
  __int64 v93; // [rsp+38h] [rbp-58h]
  __int64 v94; // [rsp+38h] [rbp-58h]
  __int64 v95; // [rsp+38h] [rbp-58h]
  __int64 v96; // [rsp+38h] [rbp-58h]
  __int64 v97; // [rsp+38h] [rbp-58h]
  __int64 v98; // [rsp+38h] [rbp-58h]
  char v99[8]; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+48h] [rbp-48h]
  char v101; // [rsp+4Ch] [rbp-44h]

  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)a2;
  v90 = sub_8D2E30(*(_QWORD *)a1);
  v92 = sub_8D2E30(v11);
  if ( dword_4F077C4 != 2 || (unsigned __int8)a4 > 0x2Cu || (v33 = 0x1007C0030000LL, !_bittest64(&v33, a4)) )
  {
    v12 = 1;
    v80 = a1 + 144;
LABEL_12:
    for ( LODWORD(v20) = v12; ; LODWORD(v20) = 0 )
    {
      if ( v90 )
      {
        v13 = a2 + 144;
        v14 = *(_BYTE *)(a2 + 16) == 2;
        if ( dword_4F077C4 != 2 && v92 )
        {
          if ( *(_BYTE *)(a1 + 16) != 2 || (v32 = sub_712570(v80), v13 = a2 + 144, !v32) )
          {
            v81 = v13;
            v15 = sub_8D46C0(v11);
            v16 = sub_8D2600(v15);
            v17 = v81;
            if ( !v16 || *(_BYTE *)(a2 + 16) == 2 && (v30 = sub_712570(v81), v17 = v81, v30) )
            {
              v79 = v17;
              v31 = sub_6EB660(a2);
              if ( (unsigned int)sub_8DFA20(
                                   v11,
                                   v14,
                                   (*(_BYTE *)(a2 + 19) & 0x10) != 0,
                                   v31,
                                   v79,
                                   v10,
                                   1,
                                   v20,
                                   42,
                                   (__int64)v99,
                                   0) )
              {
LABEL_21:
                v24 = v10;
                goto LABEL_22;
              }
            }
          }
          goto LABEL_8;
        }
        v23 = sub_6EB660(a2);
        if ( (unsigned int)sub_8DFA20(
                             v11,
                             v14,
                             (*(_BYTE *)(a2 + 19) & 0x10) != 0,
                             v23,
                             (int)a2 + 144,
                             v10,
                             1,
                             v20,
                             42,
                             (__int64)v99,
                             0) )
          goto LABEL_21;
      }
      if ( !v92 )
      {
        if ( !v20 )
          goto LABEL_16;
        goto LABEL_10;
      }
LABEL_8:
      v83 = *(_BYTE *)(a1 + 16) == 2;
      v18 = sub_6EB660(a1);
      if ( (unsigned int)sub_8DFA20(
                           v10,
                           v83,
                           (*(_BYTE *)(a1 + 19) & 0x10) != 0,
                           v18,
                           v80,
                           v11,
                           1,
                           v20,
                           42,
                           (__int64)v99,
                           0) )
      {
        if ( v90 )
        {
          v24 = v11;
LABEL_22:
          v25 = v10;
          v84 = v24;
          v26 = sub_8D46C0(v10);
          v21 = v84;
          for ( i = *(_BYTE *)(v26 + 140); i == 12; i = *(_BYTE *)(v26 + 140) )
            v26 = *(_QWORD *)(v26 + 160);
          if ( i )
          {
            if ( v92 )
              goto LABEL_80;
LABEL_26:
            v28 = 1;
            if ( !dword_4D04964 )
              goto LABEL_96;
            goto LABEL_27;
          }
        }
        else
        {
          v21 = v11;
LABEL_80:
          v25 = v11;
          v86 = v21;
          v56 = sub_8D46C0(v11);
          v21 = v86;
          for ( j = *(_BYTE *)(v56 + 140); j == 12; j = *(_BYTE *)(v56 + 140) )
            v56 = *(_QWORD *)(v56 + 160);
          if ( j )
          {
            if ( v90 )
            {
              if ( v10 != v11 )
              {
                if ( !v10
                  || !v11
                  || !dword_4F07588
                  || (v65 = *(_QWORD *)(v10 + 32), *(_QWORD *)(v11 + 32) != v65)
                  || !v65 )
                {
                  v58 = sub_8D46C0(v10);
                  v59 = sub_8D46C0(v11);
                  if ( v10 == v86
                    || v86 && v10 && dword_4F07588 && (v60 = *(_QWORD *)(v86 + 32), *(_QWORD *)(v10 + 32) == v60) && v60 )
                  {
                    v61 = sub_73CA70(v58, v59);
                  }
                  else
                  {
                    v61 = sub_73CA70(v59, v58);
                  }
                  v21 = sub_72D2E0(v61, 0);
                }
              }
              goto LABEL_26;
            }
            v28 = 0;
            if ( !dword_4D04964 )
            {
LABEL_96:
              v55 = v100;
              LODWORD(v20) = 1;
              if ( !v100 )
                goto LABEL_18;
LABEL_97:
              v97 = v21;
              LODWORD(v20) = 1;
              v62 = sub_6E53E0(5, v55, a3);
              v21 = v97;
              if ( v62 )
              {
                sub_6858D0(v100, a3, v10, v11);
                v21 = v97;
              }
              goto LABEL_18;
            }
LABEL_27:
            if ( dword_4F077C4 )
              goto LABEL_76;
            if ( a5 || (v101 & 0x10) == 0 )
            {
              k = 0;
              if ( v90 )
              {
                v88 = v21;
                v67 = sub_8D46C0(v10);
                v21 = v88;
                for ( k = v67; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                  ;
              }
              m = 0;
              if ( v92 )
              {
                v87 = v21;
                v66 = sub_8D46C0(v11);
                v21 = v87;
                for ( m = v66; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                  ;
              }
              if ( a6
                || (!v90 || (v89 = v21, v68 = sub_8D2310(k), v21 = v89, !v68))
                && (!v92 || (v85 = v21, v49 = sub_8D2310(m), v21 = v85, !v49)) )
              {
                if ( a7 )
                {
                  if ( !a8 && v28 && v92 )
                    goto LABEL_72;
                  goto LABEL_76;
                }
                if ( !v90 || (v91 = v21, v69 = sub_8D23B0(k), v21 = v91, !v69) )
                {
                  if ( !v92 )
                    goto LABEL_76;
                  v94 = v21;
                  v50 = sub_8D23B0(m);
                  v21 = v94;
                  if ( !v50 )
                  {
                    if ( a8 || !v28 )
                      goto LABEL_76;
LABEL_72:
                    v95 = v21;
                    v51 = sub_8D23B0(k);
                    v21 = v95;
                    if ( !v51 || (v52 = sub_8D2530(m), v21 = v95, !v52) )
                    {
                      v96 = v21;
                      v53 = sub_8D23B0(m);
                      v21 = v96;
                      if ( !v53 || (v54 = sub_8D2530(k), v21 = v96, !v54) )
                      {
LABEL_76:
                        v55 = v100;
                        if ( !v100 )
                          goto LABEL_77;
                        goto LABEL_97;
                      }
                    }
                  }
                }
              }
            }
            v93 = v21;
            LODWORD(v20) = 1;
            v29 = sub_6E53E0(byte_4F07472[0], 0x2Au, a3);
            v21 = v93;
            if ( v29 )
            {
              sub_686040(byte_4F07472[0], 0x2Au, a3, v10, v11);
              v21 = v93;
            }
            goto LABEL_18;
          }
        }
        v64 = sub_72C930(v25);
        LODWORD(v20) = 1;
        v21 = sub_72D2E0(v64, 0);
        goto LABEL_18;
      }
      if ( !v20 )
        goto LABEL_16;
LABEL_10:
      if ( dword_4F077C4 == 2 )
      {
        v19 = sub_8DB6D0(v10, v11);
        v12 = 0;
        if ( v19 )
          goto LABEL_17;
        goto LABEL_12;
      }
    }
  }
  if ( !(unsigned int)sub_6DF870(a1) )
  {
    LODWORD(v20) = 1;
    v63 = sub_6DF870(a2);
    v21 = v10;
    if ( v63 )
      goto LABEL_18;
LABEL_43:
    v21 = sub_8E1ED0(v10, v11, v34, v36, v35, v21);
    v20 = v21 != 0;
    if ( v21 )
      goto LABEL_18;
    if ( dword_4D04964 || v90 == 0 || (_BYTE)a4 == 44 || !v92 )
      goto LABEL_16;
    for ( n = v10; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v38 = *(_QWORD *)(n + 160);
    for ( ii = v11; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
      ;
    v40 = *(_BYTE *)(v38 + 140);
    v41 = *(_QWORD *)(ii + 160);
    v42 = v38;
    if ( v40 == 12 )
    {
      do
      {
        v42 = *(_QWORD *)(v42 + 160);
        v43 = *(_BYTE *)(v42 + 140);
      }
      while ( v43 == 12 );
    }
    else
    {
      v43 = *(_BYTE *)(v38 + 140);
    }
    v44 = *(_BYTE *)(v41 + 140);
    if ( v44 == 12 )
    {
      v45 = v41;
      do
      {
        v45 = *(_QWORD *)(v45 + 160);
        v46 = *(_BYTE *)(v45 + 140);
      }
      while ( v46 == 12 );
    }
    else
    {
      v46 = *(_BYTE *)(v41 + 140);
    }
    if ( v43 == 1 )
    {
      if ( v46 != 7 )
        goto LABEL_16;
      v70 = 0;
      if ( (v44 & 0xFB) == 8 )
        v70 = (unsigned int)sub_8D4C10(v41, dword_4F077C4 != 2);
      v71 = v38;
    }
    else
    {
      if ( v43 != 7 || v46 != 1 )
      {
LABEL_16:
        sub_6E5ED0(0x2Au, a3, v10, v11);
        v19 = sub_72C930(42);
LABEL_17:
        v21 = v19;
        goto LABEL_18;
      }
      v75 = 0;
      if ( (v40 & 0xFB) == 8 )
        v75 = sub_8D4C10(v38, dword_4F077C4 != 2);
      v70 = v75;
      v71 = v41;
    }
    v72 = sub_73C570(v71, v70, -1);
    v73 = sub_72D2E0(v72, 0);
    v21 = v73;
    if ( v73 && (v98 = v73, v74 = sub_6E53E0(5, 0xA96u, a3), v21 = v98, v74) )
    {
      LODWORD(v20) = 1;
      sub_684B30(0xA96u, a3);
      v21 = v98;
    }
    else
    {
LABEL_77:
      LODWORD(v20) = 1;
    }
    goto LABEL_18;
  }
  if ( !(unsigned int)sub_8D2660(*(_QWORD *)a2) )
  {
    v21 = v11;
    LODWORD(v20) = 1;
    if ( *(_BYTE *)(a2 + 16) != 2 )
      goto LABEL_18;
    if ( !(unsigned int)sub_712570(a2 + 144) )
    {
      v21 = v11;
      goto LABEL_18;
    }
  }
  if ( !v90 )
  {
    v21 = v11;
    LODWORD(v20) = 1;
    goto LABEL_18;
  }
  v36 = v92;
  v21 = v10;
  LODWORD(v20) = 1;
  if ( v92 )
    goto LABEL_43;
LABEL_18:
  *a9 = v21;
  return v20;
}
