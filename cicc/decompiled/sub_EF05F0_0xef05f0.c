// Function: sub_EF05F0
// Address: 0xef05f0
//
__int64 *__fastcall sub_EF05F0(unsigned __int8 **a1, char a2)
{
  unsigned __int64 v4; // rsi
  unsigned __int8 *v5; // rdx
  char *v6; // rax
  __int64 v7; // rax
  __int64 *v8; // r12
  unsigned __int8 *v9; // rax
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rcx
  unsigned __int8 *v15; // rax
  unsigned __int8 *v17; // rcx
  unsigned __int8 *v18; // rax
  int v19; // edi
  char v20; // dl
  char v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  char *v32; // rax
  unsigned __int64 *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  char *v38; // rax
  unsigned __int8 *v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  void *v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // r8
  char *v50; // rax
  __int64 v51; // r11
  char v52; // r14
  __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 *v55; // rax
  __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v61; // rax
  __int64 v62; // r8
  __int64 *v63; // rax
  __int64 v64; // rax
  char v65; // bl
  __int64 v66; // rdi
  __int64 v67; // rdi
  char v68; // dl
  char *v69; // rax
  unsigned __int64 v70; // rbx
  char *v71; // rax
  __int64 v72; // rax
  __int64 v73; // r9
  char v74; // r11
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 *v79; // r14
  int v80; // ecx
  char v81; // dl
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 *v84; // rdx
  char v85; // [rsp+Fh] [rbp-1C1h]
  __int64 v86; // [rsp+18h] [rbp-1B8h]
  char v87; // [rsp+18h] [rbp-1B8h]
  char v88; // [rsp+20h] [rbp-1B0h]
  __int64 v89; // [rsp+20h] [rbp-1B0h]
  char v90; // [rsp+20h] [rbp-1B0h]
  __int64 v91; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 *v92; // [rsp+28h] [rbp-1A8h]
  __int64 v93; // [rsp+28h] [rbp-1A8h]
  __int64 v94; // [rsp+28h] [rbp-1A8h]
  _QWORD *v95; // [rsp+28h] [rbp-1A8h]
  __int64 v96; // [rsp+28h] [rbp-1A8h]
  __int64 v97; // [rsp+28h] [rbp-1A8h]
  char v98; // [rsp+30h] [rbp-1A0h]
  __int64 v99; // [rsp+30h] [rbp-1A0h]
  __int64 v100; // [rsp+30h] [rbp-1A0h]
  __int64 v101; // [rsp+30h] [rbp-1A0h]
  __int64 v102; // [rsp+38h] [rbp-198h]
  __int64 v103; // [rsp+40h] [rbp-190h] BYREF
  __int64 *v104; // [rsp+48h] [rbp-188h] BYREF
  __int64 *v105; // [rsp+50h] [rbp-180h] BYREF
  __int64 v106; // [rsp+58h] [rbp-178h]
  unsigned __int64 v107; // [rsp+60h] [rbp-170h]
  char v108; // [rsp+68h] [rbp-168h]
  __int64 *v109; // [rsp+70h] [rbp-160h] BYREF
  __int64 v110; // [rsp+78h] [rbp-158h]
  _QWORD v111[16]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v112[26]; // [rsp+100h] [rbp-D0h] BYREF

  v4 = (unsigned __int64)a1;
  sub_EE38E0(v112, a1);
  v5 = a1[1];
  v6 = (char *)*a1;
  if ( *a1 == v5 )
    goto LABEL_4;
  if ( *v6 != 71 )
  {
    if ( *v6 == 84 )
    {
      if ( v5 - (unsigned __int8 *)v6 != 1 )
      {
        switch ( v6[1] )
        {
          case 'A':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = sub_EEF530((__int64)a1);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"template parameter object for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "template parameter object for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'C':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v70 = sub_EF1F20(a1);
            if ( !v70 )
              goto LABEL_11;
            v4 = 1;
            if ( !sub_EE32C0((char **)a1, 1) )
              goto LABEL_11;
            v71 = (char *)*a1;
            if ( *a1 == a1[1] )
              goto LABEL_11;
            if ( *v71 != 95 )
              goto LABEL_11;
            *a1 = (unsigned __int8 *)(v71 + 1);
            v72 = sub_EF1F20(a1);
            if ( !v72 )
              goto LABEL_11;
            v74 = *((_BYTE *)a1 + 937);
            v109 = v111;
            v90 = v74;
            v101 = v72;
            v110 = 0x2000000000LL;
            sub_EE40D0((__int64)&v109, 0x16u, v72, v70, v72, v73);
            v4 = (unsigned __int64)&v109;
            v8 = sub_C65B40((__int64)(a1 + 113), (__int64)&v109, (__int64 *)&v105, (__int64)off_497B2F0);
            if ( v8 )
              goto LABEL_27;
            if ( v90 )
            {
              v75 = sub_CD1D40((__int64 *)a1 + 101, 40, 3);
              *(_QWORD *)v75 = 0;
              v4 = v75;
              v8 = (__int64 *)(v75 + 8);
              *(_WORD *)(v75 + 16) = 16406;
              LOBYTE(v75) = *(_BYTE *)(v75 + 18);
              *(_QWORD *)(v4 + 24) = v101;
              *(_QWORD *)(v4 + 32) = v70;
              *(_BYTE *)(v4 + 18) = v75 & 0xF0 | 5;
              *(_QWORD *)(v4 + 8) = &unk_49DF668;
              sub_C657C0((__int64 *)a1 + 113, (__int64 *)v4, v105, (__int64)off_497B2F0);
            }
            break;
          case 'H':
            v4 = 0;
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1680(a1, 0);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"thread-local initialization routine for ";
              v8 = (__int64 *)sub_EE8500(
                                (__int64)(a1 + 101),
                                "thread-local initialization routine for ",
                                (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'I':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1F20(a1);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"typeinfo for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "typeinfo for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'S':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1F20(a1);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"typeinfo name for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "typeinfo name for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'T':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1F20(a1);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"VTT for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "VTT for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'V':
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1F20(a1);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"vtable for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "vtable for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'W':
            v4 = 0;
            *a1 = (unsigned __int8 *)(v6 + 2);
            v109 = (__int64 *)sub_EF1680(a1, 0);
            v8 = v109;
            if ( v109 )
            {
              v4 = (unsigned __int64)"thread-local wrapper routine for ";
              v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "thread-local wrapper routine for ", (__int64 *)&v109);
            }
            goto LABEL_12;
          case 'c':
            *a1 = (unsigned __int8 *)(v6 + 2);
            if ( (unsigned __int8)sub_EE3480((__int64)a1) )
              goto LABEL_11;
            if ( (unsigned __int8)sub_EE3480((__int64)a1) )
              goto LABEL_11;
            v4 = 1;
            v109 = (__int64 *)sub_EF05F0(a1, 1);
            if ( !v109 )
              goto LABEL_11;
            v4 = (unsigned __int64)"covariant return thunk to ";
            v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "covariant return thunk to ", (__int64 *)&v109);
            goto LABEL_12;
          default:
            goto LABEL_79;
        }
LABEL_91:
        if ( v109 != v111 )
          _libc_free(v109, v4);
        a1[115] = (unsigned __int8 *)v8;
        goto LABEL_12;
      }
LABEL_79:
      v65 = 0;
      *a1 = (unsigned __int8 *)(v6 + 1);
      if ( v5 != (unsigned __int8 *)(v6 + 1) )
        v65 = v6[1];
      if ( !(unsigned __int8)sub_EE3480((__int64)a1) )
      {
        v4 = 1;
        v109 = (__int64 *)sub_EF05F0(a1, 1);
        if ( v109 )
        {
          v66 = (__int64)(a1 + 101);
          if ( v65 == 118 )
          {
            v4 = (unsigned __int64)"virtual thunk to ";
            v8 = (__int64 *)sub_EE8500(v66, "virtual thunk to ", (__int64 *)&v109);
          }
          else
          {
            v4 = (unsigned __int64)"non-virtual thunk to ";
            v8 = (__int64 *)sub_EE8500(v66, "non-virtual thunk to ", (__int64 *)&v109);
          }
          goto LABEL_12;
        }
      }
      goto LABEL_11;
    }
LABEL_4:
    v4 = (unsigned __int64)&v105;
    v7 = (a1[91] - a1[90]) >> 3;
    v108 = 0;
    v105 = 0;
    v106 = 0;
    v107 = v7;
    v8 = (__int64 *)sub_EF1680(a1, &v105);
    if ( !v8 )
      goto LABEL_11;
    v9 = a1[90];
    v10 = v107;
    v11 = (a1[91] - v9) >> 3;
    if ( v107 < v11 )
    {
      v12 = v107;
      do
      {
        v14 = *(_QWORD *)&v9[8 * v12];
        v15 = a1[83];
        v4 = *(_QWORD *)(v14 + 16);
        if ( v15 == a1[84] )
          goto LABEL_11;
        v13 = *(_QWORD **)v15;
        if ( !v13 || v4 >= (__int64)(v13[1] - *v13) >> 3 )
          goto LABEL_11;
        ++v12;
        *(_QWORD *)(v14 + 24) = *(_QWORD *)(*v13 + 8 * v4);
        v9 = a1[90];
      }
      while ( v11 != v12 );
    }
    v17 = a1[1];
    a1[91] = &v9[8 * v10];
    v18 = *a1;
    if ( v17 != *a1 )
    {
      v19 = *v18;
      if ( (unsigned __int8)(v19 - 46) > 0x31u
        || (v4 = 0x2000000800001LL, !_bittest64((const __int64 *)&v4, (unsigned int)(v19 - 46))) )
      {
        if ( !a2 )
        {
          do
            *a1 = ++v18;
          while ( *(v18 - 1) && v18 != v17 );
          goto LABEL_12;
        }
        v4 = 13;
        if ( (unsigned __int8)sub_EE3B50((const void **)a1, 0xDu, "Ua9enable_ifI") )
        {
          v27 = (a1[3] - a1[2]) >> 3;
          while ( 1 )
          {
            v32 = (char *)*a1;
            if ( *a1 != a1[1] && *v32 == 69 )
              break;
            v109 = sub_EEF530((__int64)a1);
            if ( !v109 )
              goto LABEL_11;
            v4 = (unsigned __int64)&v109;
            sub_E18380((__int64)(a1 + 2), (__int64 *)&v109, v28, v29, v30, v31);
          }
          *a1 = (unsigned __int8 *)(v32 + 1);
          v33 = (unsigned __int64 *)sub_EE6060(a1, v27);
          v109 = v111;
          v92 = v33;
          v88 = *((_BYTE *)a1 + 937);
          v99 = v34;
          v110 = 0x2000000000LL;
          sub_EE4780((__int64)&v109, 0xAu, v92, v34, (__int64)v92, v34);
          v4 = (unsigned __int64)&v109;
          v35 = sub_C65B40((__int64)(a1 + 113), (__int64)&v109, (__int64 *)&v104, (__int64)off_497B2F0);
          if ( v35 )
          {
            v100 = (__int64)(v35 + 1);
            if ( v109 != v111 )
              _libc_free(v109, &v109);
            v4 = (unsigned __int64)&v109;
            v109 = (__int64 *)v100;
            v36 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v109);
            if ( v36 )
            {
              v37 = v36[1];
              if ( !v37 )
                v37 = v100;
              v100 = v37;
            }
            if ( a1[116] == (unsigned __int8 *)v100 )
              *((_BYTE *)a1 + 936) = 1;
          }
          else
          {
            if ( !v88 )
            {
              v67 = (__int64)v109;
              if ( v109 == v111 )
              {
LABEL_130:
                a1[115] = 0;
              }
              else
              {
LABEL_96:
                _libc_free(v67, &v109);
                a1[115] = 0;
              }
              goto LABEL_11;
            }
            v91 = v99;
            v82 = sub_CD1D40((__int64 *)a1 + 101, 40, 3);
            *(_QWORD *)v82 = 0;
            v4 = v82;
            v100 = v82 + 8;
            LOBYTE(v82) = *(_BYTE *)(v82 + 18);
            *(_WORD *)(v4 + 16) = 16394;
            *(_QWORD *)(v4 + 24) = v92;
            *(_QWORD *)(v4 + 32) = v91;
            *(_BYTE *)(v4 + 18) = v82 & 0xF0 | 5;
            *(_QWORD *)(v4 + 8) = &unk_49DF188;
            sub_C657C0((__int64 *)a1 + 113, (__int64 *)v4, v104, (__int64)off_497B2F0);
            if ( v109 != v111 )
              _libc_free(v109, v4);
            a1[115] = (unsigned __int8 *)v100;
          }
        }
        else
        {
          v100 = 0;
        }
        v89 = 0;
        if ( (_BYTE)v105 || !BYTE1(v105) || (v89 = sub_EF1F20(a1)) != 0 )
        {
          v38 = (char *)*a1;
          v39 = a1[1];
          if ( *a1 == v39 || *v38 != 118 )
          {
            v40 = a1[3] - a1[2];
            while ( 1 )
            {
              v41 = sub_EF1F20(a1);
              v103 = v41;
              if ( !v41 )
                goto LABEL_11;
              v45 = a1[3] - a1[2];
              if ( v45 == v40 && v108 )
              {
                v94 = v41;
                v87 = *((_BYTE *)a1 + 937);
                v110 = 0x2000000000LL;
                v109 = v111;
                sub_D953B0((__int64)&v109, 87, v45, 0x2000000000LL, v43, v44);
                sub_D953B0((__int64)&v109, v94, v57, v58, v59, v60);
                v4 = (unsigned __int64)&v109;
                v61 = sub_C65B40((__int64)(a1 + 113), (__int64)&v109, (__int64 *)&v104, (__int64)off_497B2F0);
                if ( v61 )
                {
                  v62 = (__int64)(v61 + 1);
                  if ( v109 != v111 )
                  {
                    v95 = v61 + 1;
                    _libc_free(v109, &v109);
                    v62 = (__int64)v95;
                  }
                  v109 = (__int64 *)v62;
                  v96 = v62;
                  v63 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v109);
                  v43 = v96;
                  if ( v63 )
                  {
                    v64 = v63[1];
                    if ( v64 )
                      v43 = v64;
                  }
                  if ( a1[116] == (unsigned __int8 *)v43 )
                    *((_BYTE *)a1 + 936) = 1;
                }
                else
                {
                  if ( !v87 )
                  {
                    v67 = (__int64)v109;
                    if ( v109 != v111 )
                      goto LABEL_96;
                    goto LABEL_130;
                  }
                  v76 = sub_CD1D40((__int64 *)a1 + 101, 32, 3);
                  *(_QWORD *)v76 = 0;
                  v77 = v76;
                  *(_WORD *)(v76 + 16) = 16471;
                  v97 = v76 + 8;
                  *(_BYTE *)(v76 + 18) = *(_BYTE *)(v76 + 18) & 0xF0 | 5;
                  *(_QWORD *)(v76 + 8) = &unk_49DF4E8;
                  *(_QWORD *)(v76 + 24) = v103;
                  sub_C657C0((__int64 *)a1 + 113, (__int64 *)v76, v104, (__int64)off_497B2F0);
                  v43 = v97;
                  if ( v109 != v111 )
                  {
                    _libc_free(v109, v77);
                    v43 = v97;
                  }
                  a1[115] = (unsigned __int8 *)v43;
                }
                v103 = v43;
              }
              v4 = (unsigned __int64)&v103;
              sub_E18380((__int64)(a1 + 2), &v103, v45, v42, v43, v44);
              if ( a1[1] != *a1 )
              {
                if ( (unsigned __int8)(**a1 - 46) > 0x31u )
                  continue;
                v46 = 0x2000800800001LL;
                if ( !_bittest64(&v46, (unsigned int)**a1 - 46) )
                  continue;
              }
              v4 = v40 >> 3;
              v47 = sub_EE6060(a1, v40 >> 3);
              v49 = v48;
              v50 = (char *)*a1;
              v39 = a1[1];
              goto LABEL_59;
            }
          }
          v50 = v38 + 1;
          v49 = 0;
          v47 = 0;
          *a1 = (unsigned __int8 *)v50;
LABEL_59:
          v51 = 0;
          if ( v50 == (char *)v39 )
            goto LABEL_62;
          if ( *v50 != 81 )
            goto LABEL_62;
          v52 = *((_BYTE *)a1 + 778);
          *((_BYTE *)a1 + 778) = 1;
          *a1 = (unsigned __int8 *)(v50 + 1);
          v102 = v49;
          v53 = sub_EEA9F0((__int64)a1);
          *((_BYTE *)a1 + 778) = v52;
          v49 = v102;
          v51 = v53;
          if ( v53 )
          {
LABEL_62:
            v85 = *((_BYTE *)a1 + 937);
            v110 = 0x2000000000LL;
            v109 = v111;
            v86 = v51;
            v93 = v49;
            sub_EE48D0((__int64)&v109, v89, (__int64)v8, (__int64)v47, v49, v100, v51, HIDWORD(v105), v106);
            v4 = (unsigned __int64)&v109;
            v54 = sub_C65B40((__int64)(a1 + 113), (__int64)&v109, (__int64 *)&v104, (__int64)off_497B2F0);
            if ( v54 )
            {
              v26 = (__int64)v109;
              v8 = v54 + 1;
              if ( v109 == v111 )
                goto LABEL_65;
              goto LABEL_64;
            }
            if ( v85 )
            {
              v78 = sub_CD1D40((__int64 *)a1 + 101, 80, 3);
              *(_QWORD *)v78 = 0;
              v4 = v78;
              v79 = (__int64 *)(v78 + 8);
              v80 = HIDWORD(v105);
              v81 = v106;
              *(_WORD *)(v78 + 16) = 19;
              LOBYTE(v78) = *(_BYTE *)(v78 + 18);
              *(_QWORD *)(v4 + 32) = v8;
              v8 = v79;
              *(_DWORD *)(v4 + 72) = v80;
              *(_QWORD *)(v4 + 40) = v47;
              *(_QWORD *)(v4 + 48) = v93;
              *(_BYTE *)(v4 + 18) = v78 & 0xF0 | 1;
              *(_QWORD *)(v4 + 64) = v86;
              *(_BYTE *)(v4 + 76) = v81;
              *(_QWORD *)(v4 + 8) = &unk_49DF548;
              *(_QWORD *)(v4 + 24) = v89;
              *(_QWORD *)(v4 + 56) = v100;
              sub_C657C0((__int64 *)a1 + 113, (__int64 *)v4, v104, (__int64)off_497B2F0);
            }
            else
            {
              v8 = 0;
            }
            goto LABEL_91;
          }
        }
        goto LABEL_11;
      }
    }
    goto LABEL_12;
  }
  if ( v5 - (unsigned __int8 *)v6 == 1 )
    goto LABEL_11;
  v20 = v6[1];
  if ( v20 == 82 )
  {
    v4 = 0;
    *a1 = (unsigned __int8 *)(v6 + 2);
    v105 = (__int64 *)sub_EF1680(a1, 0);
    if ( v105 )
    {
      v4 = (unsigned __int64)&v109;
      v68 = sub_EE3560((char **)a1, (__int64 *)&v109);
      v69 = (char *)*a1;
      if ( *a1 != a1[1] && *v69 == 95 )
      {
        *a1 = (unsigned __int8 *)(v69 + 1);
      }
      else if ( !v68 )
      {
        goto LABEL_11;
      }
      v4 = (unsigned __int64)"reference temporary for ";
      v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "reference temporary for ", (__int64 *)&v105);
      goto LABEL_12;
    }
LABEL_11:
    v8 = 0;
    goto LABEL_12;
  }
  if ( v20 != 86 )
  {
    if ( v20 == 73 )
    {
      v4 = (unsigned __int64)&v104;
      v104 = 0;
      *a1 = (unsigned __int8 *)(v6 + 2);
      v8 = 0;
      if ( (unsigned __int8)sub_EE9010((__int64)a1, (__int64 *)&v104) )
        goto LABEL_12;
      v8 = v104;
      if ( !v104 )
        goto LABEL_12;
      v21 = *((_BYTE *)a1 + 937);
      v109 = v111;
      v98 = v21;
      v110 = 0x2000000002LL;
      v111[0] = 21;
      sub_C653C0((__int64)&v109, "initializer for module ", 0x17u);
      sub_D953B0((__int64)&v109, (__int64)v8, v22, v23, v24, v25);
      v4 = (unsigned __int64)&v109;
      v8 = sub_C65B40((__int64)(a1 + 113), (__int64)&v109, (__int64 *)&v105, (__int64)off_497B2F0);
      if ( v8 )
      {
LABEL_27:
        v26 = (__int64)v109;
        ++v8;
        if ( v109 == v111 )
        {
LABEL_65:
          v4 = (unsigned __int64)&v109;
          v109 = v8;
          v55 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v109);
          if ( v55 )
          {
            v56 = (__int64 *)v55[1];
            if ( v56 )
              v8 = v56;
          }
          if ( a1[116] == (unsigned __int8 *)v8 )
            *((_BYTE *)a1 + 936) = 1;
          goto LABEL_12;
        }
LABEL_64:
        _libc_free(v26, &v109);
        goto LABEL_65;
      }
      if ( v98 )
      {
        v83 = sub_CD1D40((__int64 *)a1 + 101, 48, 3);
        *(_QWORD *)v83 = 0;
        v4 = v83;
        v84 = v104;
        v8 = (__int64 *)(v83 + 8);
        *(_WORD *)(v83 + 16) = 16405;
        LOBYTE(v83) = *(_BYTE *)(v83 + 18);
        *(_QWORD *)(v4 + 24) = 23;
        *(_QWORD *)(v4 + 40) = v84;
        *(_BYTE *)(v4 + 18) = v83 & 0xF0 | 5;
        *(_QWORD *)(v4 + 8) = &unk_49DF608;
        *(_QWORD *)(v4 + 32) = "initializer for module ";
        sub_C657C0((__int64 *)a1 + 113, (__int64 *)v4, v105, (__int64)off_497B2F0);
      }
      goto LABEL_91;
    }
    goto LABEL_11;
  }
  v4 = 0;
  *a1 = (unsigned __int8 *)(v6 + 2);
  v109 = (__int64 *)sub_EF1680(a1, 0);
  v8 = v109;
  if ( v109 )
  {
    v4 = (unsigned __int64)"guard variable for ";
    v8 = (__int64 *)sub_EE8500((__int64)(a1 + 101), "guard variable for ", (__int64 *)&v109);
  }
LABEL_12:
  sub_EE36A0(v112, (const void *)v4);
  return v8;
}
