// Function: sub_11572C0
// Address: 0x11572c0
//
unsigned __int8 *__fastcall sub_11572C0(__int64 a1, char a2, __int64 *a3)
{
  unsigned __int8 *v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int8 *v10; // r15
  __int64 v12; // rcx
  unsigned int v13; // r15d
  bool v14; // al
  _BOOL4 v15; // r15d
  const char *v16; // rax
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  char v24; // al
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r15
  __int64 v32; // r13
  const char *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rdx
  __int64 v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // r15
  _BYTE *v42; // rax
  unsigned __int8 *v43; // rcx
  unsigned int v44; // r15d
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rdi
  __int64 v48; // r15
  unsigned __int8 *v49; // r13
  const char *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rdx
  __int64 v53; // r12
  __int64 v54; // rbx
  __int64 v55; // r12
  __int64 v56; // rdx
  unsigned int v57; // esi
  unsigned int v58; // ebx
  unsigned __int8 *v59; // r15
  __int64 v60; // rax
  _QWORD *v61; // rdx
  _QWORD *v62; // rax
  __int64 v63; // r13
  __int64 v64; // r15
  __int64 v65; // r14
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // r15
  __int64 v71; // rdx
  unsigned int v72; // esi
  const char *v73; // rax
  _QWORD *v74; // rdx
  _QWORD *v75; // rax
  __int64 v76; // r15
  __int64 v77; // r13
  __int64 v78; // r14
  __int64 v79; // rdx
  unsigned int v80; // esi
  __int64 v81; // r15
  __int64 v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 *v85; // [rsp+8h] [rbp-C8h]
  __int64 *v86; // [rsp+10h] [rbp-C0h]
  __int64 v87; // [rsp+18h] [rbp-B8h]
  __int64 v88; // [rsp+18h] [rbp-B8h]
  _BOOL4 v89; // [rsp+18h] [rbp-B8h]
  int v90; // [rsp+18h] [rbp-B8h]
  char v91; // [rsp+20h] [rbp-B0h]
  int v92; // [rsp+20h] [rbp-B0h]
  bool v93; // [rsp+27h] [rbp-A9h]
  bool v94; // [rsp+28h] [rbp-A8h]
  __int64 *v95; // [rsp+28h] [rbp-A8h]
  __int64 v96; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+38h] [rbp-98h] BYREF
  const char *v98; // [rsp+40h] [rbp-90h] BYREF
  _QWORD *v99; // [rsp+48h] [rbp-88h] BYREF
  char *v100; // [rsp+50h] [rbp-80h]
  __int16 v101; // [rsp+60h] [rbp-70h]
  _QWORD *v102; // [rsp+70h] [rbp-60h] BYREF
  __int64 *v103; // [rsp+78h] [rbp-58h] BYREF
  __int64 *v104; // [rsp+80h] [rbp-50h]
  __int16 v105; // [rsp+90h] [rbp-40h]

  v5 = *(unsigned __int8 **)(a1 - 64);
  v6 = *(_QWORD *)(a1 - 32);
  if ( a2 )
  {
    v5 = *(unsigned __int8 **)(a1 - 32);
    v6 = *(_QWORD *)(a1 - 64);
  }
  v94 = sub_B44900(a1);
  v93 = sub_B448F0(a1);
  if ( *(_BYTE *)v6 != 54 )
    goto LABEL_4;
  v12 = *(_QWORD *)(v6 - 64);
  if ( *(_BYTE *)v12 == 17 )
  {
    v13 = *(_DWORD *)(v12 + 32);
    if ( v13 <= 0x40 )
      v14 = *(_QWORD *)(v12 + 24) == 1;
    else
      v14 = v13 - 1 == (unsigned int)sub_C444A0(v12 + 24);
  }
  else
  {
    v41 = *(_QWORD *)(v12 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 > 1 || *(_BYTE *)v12 > 0x15u )
      goto LABEL_4;
    v87 = *(_QWORD *)(v6 - 64);
    v42 = sub_AD7630(v87, 0, v7);
    v43 = (unsigned __int8 *)v87;
    if ( !v42 || *v42 != 17 )
    {
      if ( *(_BYTE *)(v41 + 8) == 17 )
      {
        v90 = *(_DWORD *)(v41 + 32);
        if ( v90 )
        {
          v91 = 0;
          v86 = a3;
          v58 = 0;
          v59 = v43;
          do
          {
            v60 = sub_AD69F0(v59, v58);
            if ( !v60 )
            {
LABEL_98:
              a3 = v86;
              goto LABEL_4;
            }
            if ( *(_BYTE *)v60 != 13 )
            {
              if ( *(_BYTE *)v60 != 17 )
                goto LABEL_98;
              if ( *(_DWORD *)(v60 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v60 + 24) != 1 )
                  goto LABEL_98;
              }
              else
              {
                v92 = *(_DWORD *)(v60 + 32);
                if ( (unsigned int)sub_C444A0(v60 + 24) != v92 - 1 )
                  goto LABEL_98;
              }
              v91 = 1;
            }
            ++v58;
          }
          while ( v90 != v58 );
          a3 = v86;
          if ( v91 )
            goto LABEL_12;
        }
      }
LABEL_4:
      v99 = 0;
      v98 = (const char *)&v97;
      v8 = *(_QWORD *)(v6 + 16);
      if ( v8 )
      {
        if ( !*(_QWORD *)(v8 + 8) && *(_BYTE *)v6 == 42 && (unsigned __int8)(**(_BYTE **)(v6 - 64) - 42) <= 0x11u )
        {
          v97 = *(_QWORD *)(v6 - 64);
          if ( (unsigned __int8)sub_993A50(&v99, *(_QWORD *)(v6 - 32)) )
          {
            v102 = 0;
            v103 = &v96;
            v45 = *(_QWORD *)(v97 + 16);
            if ( v45 )
            {
              if ( !*(_QWORD *)(v45 + 8) && *(_BYTE *)v97 == 54 )
              {
                v88 = v97;
                if ( (unsigned __int8)sub_993A50(&v102, *(_QWORD *)(v97 - 64)) )
                {
                  v46 = *(_QWORD *)(v88 - 32);
                  if ( v46 )
                  {
                    *v103 = v46;
                    if ( v94 )
                    {
                      v94 = sub_B44900(v97);
                      v89 = v94;
                    }
                    else
                    {
                      v89 = 0;
                    }
                    if ( !sub_98EF80(v5, 0, 0, 0, 0) )
                    {
                      v73 = sub_BD5D20((__int64)v5);
                      v105 = 257;
                      v98 = v73;
                      v100 = ".fr";
                      v101 = 773;
                      v99 = v74;
                      v75 = sub_BD2C40(72, unk_3F10A14);
                      v76 = (__int64)v75;
                      if ( v75 )
                        sub_B549F0((__int64)v75, (__int64)v5, (__int64)&v102, 0, 0);
                      (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a3[11]
                                                                                                + 16LL))(
                        a3[11],
                        v76,
                        &v98,
                        a3[7],
                        a3[8]);
                      v77 = *a3;
                      v78 = *a3 + 16LL * *((unsigned int *)a3 + 2);
                      if ( *a3 != v78 )
                      {
                        do
                        {
                          v79 = *(_QWORD *)(v77 + 8);
                          v80 = *(_DWORD *)v77;
                          v77 += 16;
                          sub_B99FD0(v76, v80, v79);
                        }
                        while ( v78 != v77 );
                      }
                      v5 = (unsigned __int8 *)v76;
                    }
                    v47 = a3[10];
                    v101 = 259;
                    v98 = "mulshl";
                    v48 = v96;
                    v49 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, __int64, bool, _BOOL4))(*(_QWORD *)v47 + 32LL))(
                                               v47,
                                               25,
                                               v5,
                                               v96,
                                               v93,
                                               v89);
                    if ( !v49 )
                    {
                      v105 = 257;
                      v49 = (unsigned __int8 *)sub_B504D0(25, (__int64)v5, v48, (__int64)&v102, 0, 0);
                      (*(void (__fastcall **)(__int64, unsigned __int8 *, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
                        a3[11],
                        v49,
                        &v98,
                        a3[7],
                        a3[8]);
                      if ( *a3 != *a3 + 16LL * *((unsigned int *)a3 + 2) )
                      {
                        v85 = a3;
                        v81 = *a3 + 16LL * *((unsigned int *)a3 + 2);
                        v82 = *a3;
                        do
                        {
                          v83 = *(_QWORD *)(v82 + 8);
                          v84 = *(_DWORD *)v82;
                          v82 += 16;
                          sub_B99FD0((__int64)v49, v84, v83);
                        }
                        while ( v81 != v82 );
                        a3 = v85;
                      }
                      if ( v93 )
                        sub_B447F0(v49, 1);
                      if ( v94 )
                        sub_B44850(v49, 1);
                    }
                    v50 = sub_BD5D20(a1);
                    v51 = a3[10];
                    v99 = v52;
                    v101 = 261;
                    v98 = v50;
                    v10 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, unsigned __int8 *, bool, _BOOL4))(*(_QWORD *)v51 + 32LL))(
                                               v51,
                                               13,
                                               v49,
                                               v5,
                                               v93,
                                               v89);
                    if ( !v10 )
                    {
                      v105 = 257;
                      v10 = (unsigned __int8 *)sub_B504D0(13, (__int64)v49, (__int64)v5, (__int64)&v102, 0, 0);
                      (*(void (__fastcall **)(__int64, unsigned __int8 *, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
                        a3[11],
                        v10,
                        &v98,
                        a3[7],
                        a3[8]);
                      v53 = 16LL * *((unsigned int *)a3 + 2);
                      v54 = *a3;
                      v55 = v54 + v53;
                      while ( v55 != v54 )
                      {
                        v56 = *(_QWORD *)(v54 + 8);
                        v57 = *(_DWORD *)v54;
                        v54 += 16;
                        sub_B99FD0((__int64)v10, v57, v56);
                      }
LABEL_18:
                      if ( v93 )
                        sub_B447F0(v10, 1);
                      if ( v94 )
                        sub_B44850(v10, 1);
                      return v10;
                    }
                    return v10;
                  }
                }
              }
            }
          }
        }
      }
      v102 = 0;
      v103 = 0;
      v104 = &v96;
      v9 = *(_QWORD *)(v6 + 16);
      if ( v9 && !*(_QWORD *)(v9 + 8) && *(_BYTE *)v6 == 59 )
      {
        v24 = sub_995B10(&v102, *(_QWORD *)(v6 - 64));
        v25 = *(_QWORD *)(v6 - 32);
        if ( v24 )
        {
          v26 = *(_QWORD *)(v25 + 16);
          if ( v26 )
          {
            if ( !*(_QWORD *)(v26 + 8) && *(_BYTE *)v25 == 54 )
            {
              if ( (unsigned __int8)sub_995B10(&v103, *(_QWORD *)(v25 - 64)) )
              {
                v29 = *(_QWORD *)(v25 - 32);
                if ( v29 )
                  goto LABEL_32;
              }
              v25 = *(_QWORD *)(v6 - 32);
            }
          }
        }
        if ( (unsigned __int8)sub_995B10(&v102, v25) )
        {
          v27 = *(_QWORD *)(v6 - 64);
          v28 = *(_QWORD *)(v27 + 16);
          if ( v28 )
          {
            if ( !*(_QWORD *)(v28 + 8) && *(_BYTE *)v27 == 54 )
            {
              if ( (unsigned __int8)sub_995B10(&v103, *(_QWORD *)(v27 - 64)) )
              {
                v29 = *(_QWORD *)(v27 - 32);
                if ( v29 )
                {
LABEL_32:
                  *v104 = v29;
                  if ( !sub_98EF80(v5, 0, 0, 0, 0) )
                  {
                    v98 = sub_BD5D20((__int64)v5);
                    v100 = ".fr";
                    v101 = 773;
                    v99 = v61;
                    v105 = 257;
                    v62 = sub_BD2C40(72, unk_3F10A14);
                    v63 = (__int64)v62;
                    if ( v62 )
                      sub_B549F0((__int64)v62, (__int64)v5, (__int64)&v102, 0, 0);
                    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
                      a3[11],
                      v63,
                      &v98,
                      a3[7],
                      a3[8]);
                    v64 = *a3;
                    v65 = *a3 + 16LL * *((unsigned int *)a3 + 2);
                    if ( *a3 != v65 )
                    {
                      do
                      {
                        v66 = *(_QWORD *)(v64 + 8);
                        v67 = *(_DWORD *)v64;
                        v64 += 16;
                        sub_B99FD0(v63, v67, v66);
                      }
                      while ( v65 != v64 );
                    }
                    v5 = (unsigned __int8 *)v63;
                  }
                  v30 = a3[10];
                  v101 = 259;
                  v98 = "mulshl";
                  v31 = v96;
                  v32 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v30 + 32LL))(
                          v30,
                          25,
                          v5,
                          v96,
                          0,
                          0);
                  if ( !v32 )
                  {
                    v105 = 257;
                    v32 = sub_B504D0(25, (__int64)v5, v31, (__int64)&v102, 0, 0);
                    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
                      a3[11],
                      v32,
                      &v98,
                      a3[7],
                      a3[8]);
                    v68 = *a3 + 16LL * *((unsigned int *)a3 + 2);
                    if ( *a3 != v68 )
                    {
                      v95 = a3;
                      v69 = *a3;
                      v70 = v68;
                      do
                      {
                        v71 = *(_QWORD *)(v69 + 8);
                        v72 = *(_DWORD *)v69;
                        v69 += 16;
                        sub_B99FD0(v32, v72, v71);
                      }
                      while ( v70 != v69 );
                      a3 = v95;
                    }
                  }
                  v33 = sub_BD5D20(a1);
                  v34 = a3[10];
                  v99 = v35;
                  v101 = 261;
                  v98 = v33;
                  v10 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD, _QWORD))(*(_QWORD *)v34 + 32LL))(
                                             v34,
                                             15,
                                             v32,
                                             v5,
                                             0,
                                             0);
                  if ( !v10 )
                  {
                    v105 = 257;
                    v10 = (unsigned __int8 *)sub_B504D0(15, v32, (__int64)v5, (__int64)&v102, 0, 0);
                    (*(void (__fastcall **)(__int64, unsigned __int8 *, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
                      a3[11],
                      v10,
                      &v98,
                      a3[7],
                      a3[8]);
                    v36 = 16LL * *((unsigned int *)a3 + 2);
                    v37 = *a3;
                    v38 = v37 + v36;
                    while ( v38 != v37 )
                    {
                      v39 = *(_QWORD *)(v37 + 8);
                      v40 = *(_DWORD *)v37;
                      v37 += 16;
                      sub_B99FD0((__int64)v10, v40, v39);
                    }
                  }
                  return v10;
                }
              }
            }
          }
        }
      }
      return 0;
    }
    v44 = *((_DWORD *)v42 + 8);
    if ( v44 <= 0x40 )
    {
      if ( *((_QWORD *)v42 + 3) != 1 )
        goto LABEL_4;
      goto LABEL_12;
    }
    v14 = v44 - 1 == (unsigned int)sub_C444A0((__int64)(v42 + 24));
  }
  if ( !v14 )
    goto LABEL_4;
LABEL_12:
  if ( !*(_QWORD *)(v6 - 32) )
    goto LABEL_4;
  v15 = 0;
  v96 = *(_QWORD *)(v6 - 32);
  if ( v94 )
  {
    v15 = (*(_BYTE *)(v6 + 1) & 4) != 0;
    v94 = (*(_BYTE *)(v6 + 1) & 4) != 0;
  }
  v16 = sub_BD5D20(a1);
  v17 = a3[10];
  v101 = 261;
  v99 = v18;
  v98 = v16;
  v10 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, __int64, bool, _BOOL4))(*(_QWORD *)v17 + 32LL))(
                             v17,
                             25,
                             v5,
                             v96,
                             v93,
                             v15);
  if ( !v10 )
  {
    v105 = 257;
    v10 = (unsigned __int8 *)sub_B504D0(25, (__int64)v5, v96, (__int64)&v102, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, const char **, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v10,
      &v98,
      a3[7],
      a3[8]);
    v19 = 16LL * *((unsigned int *)a3 + 2);
    v20 = *a3;
    v21 = v20 + v19;
    while ( v21 != v20 )
    {
      v22 = *(_QWORD *)(v20 + 8);
      v23 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0((__int64)v10, v23, v22);
    }
    goto LABEL_18;
  }
  return v10;
}
