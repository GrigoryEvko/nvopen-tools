// Function: sub_1209C30
// Address: 0x1209c30
//
char __fastcall sub_1209C30(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // r8
  __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 k; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r12
  __int64 v15; // r15
  _QWORD *v16; // r8
  _QWORD *v17; // r12
  __int64 v18; // r14
  __int64 v19; // r14
  __int64 v20; // r13
  __int64 i; // r10
  __int64 v22; // rax
  __int64 v23; // r9
  _QWORD *v24; // r11
  __int64 v25; // r10
  __int64 v26; // rax
  int v27; // eax
  int v28; // edx
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // r12
  _QWORD *j; // r14
  _BYTE *v37; // rax
  _BYTE *v38; // rcx
  _BYTE *v39; // r8
  __int64 v40; // rdx
  unsigned __int64 v41; // r8
  unsigned int v42; // eax
  __int64 *v43; // r12
  __int64 v44; // r14
  __int64 *v45; // rbx
  __int64 v46; // r13
  __int64 v47; // rdi
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rdi
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // r8
  __int64 v60; // [rsp+18h] [rbp-B8h]
  __int64 v61; // [rsp+20h] [rbp-B0h]
  __int64 v62; // [rsp+20h] [rbp-B0h]
  _QWORD *v63; // [rsp+20h] [rbp-B0h]
  __int64 v64; // [rsp+28h] [rbp-A8h]
  int v65; // [rsp+28h] [rbp-A8h]
  __int64 v66; // [rsp+28h] [rbp-A8h]
  __int64 v67; // [rsp+28h] [rbp-A8h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  __int64 v69; // [rsp+30h] [rbp-A0h]
  char v70; // [rsp+30h] [rbp-A0h]
  __int64 v71; // [rsp+38h] [rbp-98h]
  __int64 v72; // [rsp+38h] [rbp-98h]
  __int64 v73; // [rsp+38h] [rbp-98h]
  __int64 v74; // [rsp+38h] [rbp-98h]
  __int64 v75; // [rsp+38h] [rbp-98h]
  _QWORD *v76; // [rsp+38h] [rbp-98h]
  __int64 v77; // [rsp+38h] [rbp-98h]
  __int64 v78; // [rsp+38h] [rbp-98h]
  _BYTE *v79; // [rsp+38h] [rbp-98h]
  _QWORD *v80; // [rsp+40h] [rbp-90h]
  __int64 v81; // [rsp+40h] [rbp-90h]
  __int64 v82; // [rsp+48h] [rbp-88h]
  __int64 v83; // [rsp+48h] [rbp-88h]
  char v84; // [rsp+5Fh] [rbp-71h] BYREF
  _BYTE *v85; // [rsp+60h] [rbp-70h] BYREF
  __int64 v86; // [rsp+68h] [rbp-68h]
  _BYTE v87[96]; // [rsp+70h] [rbp-60h] BYREF

  v1 = a1[43];
  v2 = v1 + 24;
  v60 = v1 + 24;
  v69 = *(_QWORD *)(v1 + 32);
  if ( v69 != v1 + 24 )
  {
    do
    {
      if ( !v69 )
      {
        sub_B98130(0, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_1205510, (__int64)&v84);
        BUG();
      }
      v2 = (__int64)sub_1205510;
      sub_B98130(v69 - 56, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_1205510, (__int64)&v84);
      v3 = v69 + 16;
      v4 = *(_QWORD *)(v69 + 24);
      if ( v69 + 16 != v4 )
      {
        if ( !v4 )
          BUG();
        while ( *(_QWORD *)(v4 + 32) == v4 + 24 )
        {
          v4 = *(_QWORD *)(v4 + 8);
          if ( v3 == v4 )
            goto LABEL_9;
          if ( !v4 )
            BUG();
        }
        if ( v3 != v4 )
        {
          v19 = *(_QWORD *)(v4 + 32);
          v20 = v69 + 16;
          while ( 1 )
          {
            for ( i = *(_QWORD *)(v19 + 8); ; i = *(_QWORD *)(v4 + 32) )
            {
              v22 = v4 - 24;
              if ( !v4 )
                v22 = 0;
              if ( i != v22 + 48 )
                break;
              v4 = *(_QWORD *)(v4 + 8);
              if ( v20 == v4 )
              {
                v2 = (__int64)sub_1205510;
                v73 = i;
                sub_B984B0(
                  v19 - 24,
                  (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_1205510,
                  (__int64)&v84);
                v24 = (_QWORD *)(v19 - 24);
                v25 = v73;
                if ( *(_BYTE *)(v19 - 24) == 85 )
                {
                  v26 = *(_QWORD *)(v19 - 56);
                  if ( v26 )
                  {
                    if ( !*(_BYTE *)v26 )
                      goto LABEL_55;
                  }
                }
                goto LABEL_9;
              }
              if ( !v4 )
                BUG();
            }
            v2 = (__int64)sub_1205510;
            v72 = i;
            sub_B984B0(v19 - 24, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_1205510, (__int64)&v84);
            v24 = (_QWORD *)(v19 - 24);
            v25 = v72;
            if ( *(_BYTE *)(v19 - 24) == 85 )
            {
              v26 = *(_QWORD *)(v19 - 56);
              if ( v26 )
              {
                if ( !*(_BYTE *)v26 )
                {
LABEL_55:
                  if ( *(_QWORD *)(v26 + 24) == *(_QWORD *)(v19 + 56) && (*(_BYTE *)(v26 + 33) & 0x20) != 0 )
                  {
                    v27 = *(_DWORD *)(v26 + 36);
                    if ( (unsigned int)(v27 - 68) <= 3 || v27 == 155 )
                    {
                      v85 = v87;
                      v86 = 0x600000000LL;
                      v28 = *(unsigned __int8 *)(v19 - 24);
                      if ( v28 == 40 )
                      {
                        v66 = v25;
                        v76 = v24;
                        v42 = sub_B491D0((__int64)v24);
                        v24 = v76;
                        v25 = v66;
                        v29 = -32 - 32LL * v42;
                      }
                      else
                      {
                        v29 = -32;
                        if ( v28 != 85 )
                        {
                          if ( v28 != 34 )
                            BUG();
                          v29 = -96;
                        }
                      }
                      if ( *(char *)(v19 - 17) < 0 )
                      {
                        v61 = v25;
                        v64 = (__int64)v24;
                        v30 = sub_BD2BC0((__int64)v24);
                        v24 = (_QWORD *)v64;
                        v25 = v61;
                        v74 = v31 + v30;
                        if ( *(char *)(v19 - 17) >= 0 )
                        {
                          if ( (unsigned int)(v74 >> 4) )
LABEL_130:
                            BUG();
                        }
                        else
                        {
                          v32 = sub_BD2BC0(v64);
                          v24 = (_QWORD *)v64;
                          v25 = v61;
                          if ( (unsigned int)((v74 - v32) >> 4) )
                          {
                            v75 = v61;
                            if ( *(char *)(v19 - 17) >= 0 )
                              goto LABEL_130;
                            v62 = v64;
                            v65 = *(_DWORD *)(sub_BD2BC0(v64) + 8);
                            if ( *(char *)(v19 - 17) >= 0 )
                              BUG();
                            v33 = sub_BD2BC0(v62);
                            v25 = v75;
                            v24 = (_QWORD *)v62;
                            v29 -= 32LL * (unsigned int)(*(_DWORD *)(v33 + v34 - 4) - v65);
                          }
                        }
                      }
                      v35 = (_QWORD *)((char *)v24 + v29);
                      v2 = (unsigned int)v86;
                      for ( j = &v24[-4 * (*(_DWORD *)(v19 - 20) & 0x7FFFFFF)]; v35 != j; j += 4 )
                      {
                        v37 = (_BYTE *)*j;
                        if ( *(_BYTE *)*j == 24 )
                        {
                          v38 = (_BYTE *)*((_QWORD *)v37 + 3);
                          if ( (unsigned __int8)(*v38 - 5) <= 0x1Fu && (v38[1] & 0x7F) == 2 )
                          {
                            v40 = (unsigned int)v2;
                            v41 = (unsigned int)v2 + 1LL;
                            if ( v41 > HIDWORD(v86) )
                            {
                              v63 = v24;
                              v68 = v25;
                              v79 = (_BYTE *)*j;
                              sub_C8D5F0((__int64)&v85, v87, v41, 8u, v41, v23);
                              v40 = (unsigned int)v86;
                              v24 = v63;
                              v25 = v68;
                              v37 = v79;
                            }
                            *(_QWORD *)&v85[8 * v40] = v37;
                            v2 = (unsigned int)(v86 + 1);
                            LODWORD(v86) = v86 + 1;
                          }
                        }
                      }
                      v39 = v85;
                      if ( (_DWORD)v2 )
                      {
                        v77 = v25;
                        sub_B43D60(v24);
                        v43 = (__int64 *)v85;
                        v25 = v77;
                        v39 = &v85[8 * (unsigned int)v86];
                        if ( v85 != v39 )
                        {
                          v67 = v4;
                          v44 = v77;
                          v45 = (__int64 *)&v85[8 * (unsigned int)v86];
                          v78 = v20;
                          do
                          {
                            while ( 1 )
                            {
                              v46 = *v43;
                              if ( !*(_QWORD *)(*v43 + 16) )
                                break;
                              if ( v45 == ++v43 )
                                goto LABEL_89;
                            }
                            v47 = *v43++;
                            sub_B91290(v47);
                            v2 = 32;
                            j_j___libc_free_0(v46, 32);
                          }
                          while ( v45 != v43 );
LABEL_89:
                          v20 = v78;
                          v4 = v67;
                          v25 = v44;
                          v39 = v85;
                        }
                      }
                      if ( v39 != v87 )
                      {
                        v81 = v25;
                        _libc_free(v39, v2);
                        v25 = v81;
                      }
                    }
                  }
                }
              }
            }
            if ( v20 == v4 )
              break;
            v19 = v25;
          }
        }
      }
LABEL_9:
      v69 = *(_QWORD *)(v69 + 8);
    }
    while ( v60 != v69 );
    v1 = a1[43];
  }
  v5 = *(_QWORD *)(v1 + 16);
  for ( k = v1 + 8; k != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v7 = v5 - 56;
    v2 = (__int64)sub_1205510;
    if ( !v5 )
      v7 = 0;
    sub_B98130(v7, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_1205510, (__int64)&v84);
  }
  LOBYTE(v8) = (_BYTE)a1;
  v9 = a1[134];
  v10 = (__int64)(a1 + 132);
  if ( a1 + 132 != (_QWORD *)v9 )
  {
    v80 = a1 + 126;
    do
    {
      v11 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL);
      v70 = (v11 >> 2) & 1;
      if ( ((v11 >> 2) & 1) == 0 )
        BUG();
      v71 = sub_220EEE0(v9);
      LODWORD(v8) = *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) >> 1;
      if ( (_DWORD)v8 != 1 )
        goto LABEL_17;
      if ( !a1[127] )
      {
        LOBYTE(v8) = (v11 >> 2) & 1;
        v14 = a1 + 126;
        goto LABEL_27;
      }
      v13 = *(unsigned int *)(v9 + 32);
      v14 = a1 + 126;
      v15 = a1[127];
      while ( 1 )
      {
        while ( *(_DWORD *)(v15 + 32) < (unsigned int)v13 )
        {
          v15 = *(_QWORD *)(v15 + 24);
          if ( !v15 )
            goto LABEL_26;
        }
        v8 = *(_QWORD *)(v15 + 16);
        if ( *(_DWORD *)(v15 + 32) <= (unsigned int)v13 )
          break;
        v14 = (_QWORD *)v15;
        v15 = *(_QWORD *)(v15 + 16);
        if ( !v8 )
        {
LABEL_26:
          LOBYTE(v8) = v80 == v14;
LABEL_27:
          if ( (_QWORD *)a1[128] == v14 && (_BYTE)v8 )
          {
LABEL_29:
            sub_1206DC0((_QWORD *)a1[127]);
            LOBYTE(v8) = (_BYTE)a1 - 16;
            a1[127] = 0;
            a1[130] = 0;
            a1[128] = v80;
            a1[129] = v80;
          }
          goto LABEL_30;
        }
      }
      v2 = *(_QWORD *)(v15 + 24);
      while ( v2 )
      {
        if ( (unsigned int)v13 >= *(_DWORD *)(v2 + 32) )
        {
          v2 = *(_QWORD *)(v2 + 24);
        }
        else
        {
          v14 = (_QWORD *)v2;
          v2 = *(_QWORD *)(v2 + 16);
        }
      }
      while ( v8 )
      {
        while ( 1 )
        {
          v2 = *(_QWORD *)(v8 + 24);
          if ( (unsigned int)v13 <= *(_DWORD *)(v8 + 32) )
            break;
          v8 = *(_QWORD *)(v8 + 24);
          if ( !v2 )
            goto LABEL_112;
        }
        v15 = v8;
        v8 = *(_QWORD *)(v8 + 16);
      }
LABEL_112:
      if ( a1[128] != v15 )
        goto LABEL_118;
      if ( v80 == v14 )
        goto LABEL_29;
      while ( (_QWORD *)v15 != v14 )
      {
        v82 = v15;
        v15 = sub_220EF30(v15);
        v55 = sub_220F330(v82, v80);
        v56 = *(_QWORD *)(v55 + 40);
        v57 = v55;
        if ( v56 )
        {
          v83 = v55;
          sub_B91220(v55 + 40, v56);
          v57 = v83;
        }
        v2 = 48;
        LOBYTE(v8) = j_j___libc_free_0(v57, 48);
        --a1[130];
LABEL_118:
        ;
      }
LABEL_30:
      v16 = (_QWORD *)a1[133];
      if ( v16 )
      {
        v12 = *(unsigned int *)(v9 + 32);
        v17 = a1 + 132;
        v18 = a1[133];
        while ( 1 )
        {
          while ( *(_DWORD *)(v18 + 32) < (unsigned int)v12 )
          {
            v18 = *(_QWORD *)(v18 + 24);
            if ( !v18 )
              goto LABEL_36;
          }
          v8 = *(_QWORD *)(v18 + 16);
          if ( *(_DWORD *)(v18 + 32) <= (unsigned int)v12 )
            break;
          v17 = (_QWORD *)v18;
          v18 = *(_QWORD *)(v18 + 16);
          if ( !v8 )
          {
LABEL_36:
            v70 = v10 == (_QWORD)v17;
            goto LABEL_37;
          }
        }
        v13 = *(_QWORD *)(v18 + 24);
        while ( v13 )
        {
          v2 = *(_QWORD *)(v13 + 24);
          if ( (unsigned int)v12 >= *(_DWORD *)(v13 + 32) )
          {
            v13 = *(_QWORD *)(v13 + 24);
          }
          else
          {
            v17 = (_QWORD *)v13;
            v13 = *(_QWORD *)(v13 + 16);
          }
        }
        while ( v8 )
        {
          while ( 1 )
          {
            v2 = *(_QWORD *)(v8 + 16);
            v13 = *(_QWORD *)(v8 + 24);
            if ( (unsigned int)v12 <= *(_DWORD *)(v8 + 32) )
              break;
            v8 = *(_QWORD *)(v8 + 24);
            if ( !v13 )
              goto LABEL_97;
          }
          v18 = v8;
          v8 = *(_QWORD *)(v8 + 16);
        }
LABEL_97:
        if ( a1[134] != v18 || (_QWORD *)v10 != v17 )
        {
          for ( ; (_QWORD *)v18 != v17; --a1[136] )
          {
            v48 = v18;
            v18 = sub_220EF30(v18);
            v49 = sub_220F330(v48, v10);
            v53 = *(_QWORD *)(v49 + 40);
            v54 = v49;
            if ( v53 )
              sub_BA65D0(v53, v10, v50, v51, v52);
            v2 = 56;
            LOBYTE(v8) = j_j___libc_free_0(v54, 56);
          }
          goto LABEL_17;
        }
      }
      else
      {
        v17 = a1 + 132;
LABEL_37:
        if ( (_QWORD *)a1[134] != v17 || !v70 )
          goto LABEL_17;
      }
      LOBYTE(v8) = sub_1206350(v16, v2, v12, v13, (__int64)v16);
      a1[134] = v10;
      a1[133] = 0;
      a1[135] = v10;
      a1[136] = 0;
LABEL_17:
      v9 = v71;
    }
    while ( v10 != v71 );
  }
  return v8;
}
