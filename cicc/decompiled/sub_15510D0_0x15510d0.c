// Function: sub_15510D0
// Address: 0x15510d0
//
void __fastcall sub_15510D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 *v6; // r14
  char v7; // al
  _QWORD *v8; // rax
  const char *v9; // rsi
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r13
  unsigned __int8 v17; // si
  const char *v18; // rax
  char v19; // al
  _QWORD *v20; // rdi
  __int64 *v21; // rdi
  const char **v22; // rax
  unsigned int j; // r13d
  __int64 v27; // rax
  unsigned __int8 v28; // si
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // r13
  __int64 v34; // rsi
  __int64 v35; // r15
  __int64 v36; // r12
  __int64 v37; // rsi
  __int64 i; // r14
  const char *v39; // rax
  const char *v40; // rax
  __int64 v41; // rax
  unsigned int m; // ebx
  __int64 v43; // rsi
  __int64 v44; // rax
  const char *v45; // rax
  const char *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 **v52; // rbx
  __int64 *v53; // rax
  __int64 *v54; // r13
  __int64 v55; // r14
  __int64 v56; // rbx
  __int64 v57; // rdx
  __int64 v58; // rax
  unsigned int k; // ebx
  __int64 v60; // rsi
  __int64 v61; // rax
  unsigned int v62; // eax
  const char *v63; // rax
  size_t v64; // rdx
  int v65; // edx
  __int64 v66; // rcx
  __int64 v67; // r13
  __int64 v68; // rax
  __int64 v70; // [rsp+10h] [rbp-150h]
  __int64 v71; // [rsp+18h] [rbp-148h]
  __int64 v72; // [rsp+20h] [rbp-140h]
  __int64 v73; // [rsp+28h] [rbp-138h]
  __int64 v74; // [rsp+28h] [rbp-138h]
  __int64 v75; // [rsp+30h] [rbp-130h]
  double v76; // [rsp+40h] [rbp-120h]
  int v77; // [rsp+48h] [rbp-118h]
  double v78; // [rsp+48h] [rbp-118h]
  __int64 v79; // [rsp+48h] [rbp-118h]
  __int64 v80; // [rsp+48h] [rbp-118h]
  __int64 v81; // [rsp+48h] [rbp-118h]
  __int64 v82; // [rsp+50h] [rbp-110h]
  int v83; // [rsp+50h] [rbp-110h]
  int v84; // [rsp+50h] [rbp-110h]
  __int64 v85; // [rsp+50h] [rbp-110h]
  __int64 v86; // [rsp+50h] [rbp-110h]
  int v87; // [rsp+50h] [rbp-110h]
  _QWORD *v88; // [rsp+58h] [rbp-108h]
  __int64 v89; // [rsp+58h] [rbp-108h]
  unsigned int v90; // [rsp+58h] [rbp-108h]
  __int64 v91; // [rsp+58h] [rbp-108h]
  bool v92; // [rsp+58h] [rbp-108h]
  __int64 v93; // [rsp+58h] [rbp-108h]
  __int64 v94; // [rsp+58h] [rbp-108h]
  __int64 v95; // [rsp+58h] [rbp-108h]
  char v96; // [rsp+6Fh] [rbp-F1h] BYREF
  const char **v97; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v98; // [rsp+78h] [rbp-E8h]
  const char **v99; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v100; // [rsp+88h] [rbp-D8h] BYREF
  __int64 v101; // [rsp+90h] [rbp-D0h]
  const char *v102; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v103; // [rsp+A8h] [rbp-B8h] BYREF
  int v104; // [rsp+B0h] [rbp-B0h] BYREF
  __int16 v105; // [rsp+B4h] [rbp-ACh]
  char v106; // [rsp+B6h] [rbp-AAh]

  v5 = a1;
  v6 = a2;
  v7 = *((_BYTE *)a2 + 16);
  if ( v7 != 13 )
  {
    if ( v7 == 14 )
    {
      v10 = a2[4];
      v82 = sub_1698270(a1, a2);
      v11 = sub_1698280(a1);
      v14 = sub_16982C0(a1, a2, v12, v13);
      v88 = a2 + 4;
      v15 = v14;
      if ( v82 != v10 && v10 != v11 )
      {
        sub_1263B40(a1, "0x");
        if ( a2[4] == v15 )
          sub_169D930(&v97, v88);
        else
          sub_169D7E0(&v97, v88);
        v16 = a2[4];
        if ( v16 == sub_16982A0() )
        {
          sub_1549FC0(a1, 0x4Bu);
          sub_16A8130(&v99, &v97, 16);
          v45 = (const char *)v99;
          if ( (unsigned int)v100 > 0x40 )
            v45 = *v99;
          v102 = v45;
          v105 = 257;
          v103 = 0;
          v104 = 4;
          v106 = 0;
          sub_16E87C0(a1, &v102);
          if ( (unsigned int)v100 > 0x40 && v99 )
            j_j___libc_free_0_0(v99);
          sub_16A88B0(&v99, &v97, 64);
          v40 = (const char *)v99;
          if ( (unsigned int)v100 <= 0x40 )
            goto LABEL_95;
        }
        else
        {
          v17 = 76;
          if ( v16 != sub_1698290() )
          {
            if ( v16 != v15 )
            {
              sub_1549FC0(a1, 0x48u);
              v18 = (const char *)v97;
              if ( v98 > 0x40 )
                v18 = *v97;
              v102 = v18;
              v103 = 0;
              v104 = 4;
              v105 = 257;
              v106 = 0;
              sub_16E87C0(a1, &v102);
LABEL_19:
              if ( v98 > 0x40 && v97 )
                j_j___libc_free_0_0(v97);
              return;
            }
            v17 = 77;
          }
          sub_1549FC0(a1, v17);
          sub_16A88B0(&v99, &v97, 64);
          v39 = (const char *)v99;
          if ( (unsigned int)v100 > 0x40 )
            v39 = *v99;
          v102 = v39;
          v105 = 257;
          v103 = 0;
          v104 = 16;
          v106 = 0;
          sub_16E87C0(a1, &v102);
          if ( (unsigned int)v100 > 0x40 && v99 )
            j_j___libc_free_0_0(v99);
          sub_16A8130(&v99, &v97, 64);
          v40 = (const char *)v99;
          if ( (unsigned int)v100 <= 0x40 )
          {
LABEL_95:
            v102 = v40;
            v103 = 0;
            v104 = 16;
            v105 = 257;
            v106 = 0;
            sub_16E87C0(a1, &v102);
            if ( (unsigned int)v100 > 0x40 && v99 )
              j_j___libc_free_0_0(v99);
            goto LABEL_19;
          }
        }
        v40 = *(const char **)v40;
        goto LABEL_95;
      }
      if ( v10 == v14 )
      {
        v29 = a2[5];
        v20 = (_QWORD *)(v29 + 8);
        if ( (*(_BYTE *)(v29 + 26) & 7u) <= 1 )
          goto LABEL_60;
      }
      else
      {
        v19 = *((_BYTE *)a2 + 50) & 7;
        if ( v19 == 1 || (v20 = a2 + 4, !v19) )
        {
LABEL_38:
          sub_16986C0(&v103, v88);
LABEL_39:
          if ( v10 != v11 )
            sub_16A3360(&v102, v11, 0, &v96);
          if ( v15 == v103 )
            sub_169D930(&v97, &v103);
          else
            sub_169D7E0(&v97, &v103);
          v22 = v97;
          if ( v98 > 0x40 )
            v22 = (const char **)*v97;
          v99 = v22;
          WORD2(v101) = 257;
          v100 = 0;
          LODWORD(v101) = 0;
          BYTE6(v101) = 1;
          sub_16E87C0(v5, &v99);
          if ( v98 > 0x40 && v97 )
            j_j___libc_free_0_0(v97);
          sub_127D120(&v103);
          return;
        }
      }
      if ( v10 == v11 )
        v76 = sub_169D8E0(v20);
      else
        v76 = sub_169D890(v20);
      v102 = (const char *)&v104;
      v103 = 0x8000000000LL;
      if ( a2[4] == v15 )
        sub_16A4A90(v88, &v102, 6, 0, 0);
      else
        sub_16A3760(v88, &v102, 6, 0, 0);
      sub_169E660(&v99, v11, v102, (unsigned int)v103);
      v21 = &v100;
      if ( v100 == v15 )
        v21 = (__int64 *)(v101 + 8);
      v78 = sub_169D8E0(v21);
      if ( v100 == v15 )
      {
        v75 = v101;
        if ( v101 )
        {
          v73 = *(_QWORD *)(v101 - 8);
          if ( v101 != v101 + 32 * v73 )
          {
            v71 = v11;
            v32 = v101 + 32 * v73;
            v72 = v10;
            v70 = v5;
            do
            {
              v32 -= 32;
              if ( *(_QWORD *)(v32 + 8) == v15 )
              {
                v33 = *(_QWORD *)(v32 + 16);
                if ( v33 )
                {
                  v34 = 32LL * *(_QWORD *)(v33 - 8);
                  v35 = v33 + v34;
                  if ( v33 != v33 + v34 )
                  {
                    v74 = v32;
                    do
                    {
                      v35 -= 32;
                      if ( *(_QWORD *)(v35 + 8) == v15 )
                      {
                        v36 = *(_QWORD *)(v35 + 16);
                        if ( v36 )
                        {
                          v37 = 32LL * *(_QWORD *)(v36 - 8);
                          for ( i = v36 + v37; v36 != i; sub_127D120((_QWORD *)(i + 8)) )
                            i -= 32;
                          j_j_j___libc_free_0_0(v36 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v35 + 8);
                      }
                    }
                    while ( v33 != v35 );
                    v32 = v74;
                  }
                  j_j_j___libc_free_0_0(v33 - 8);
                }
              }
              else
              {
                sub_1698460(v32 + 8);
              }
            }
            while ( v75 != v32 );
            v10 = v72;
            v11 = v71;
            v5 = v70;
            v6 = a2;
          }
          j_j_j___libc_free_0_0(v75 - 8);
        }
      }
      else
      {
        sub_1698460(&v100);
      }
      if ( v76 == v78 )
      {
        sub_16E7EE0(v5, v102, (unsigned int)v103);
        if ( v102 != (const char *)&v104 )
          _libc_free((unsigned __int64)v102);
        return;
      }
      if ( v102 != (const char *)&v104 )
        _libc_free((unsigned __int64)v102);
      if ( v6[4] != v15 )
        goto LABEL_38;
LABEL_60:
      sub_169C6E0(&v103, v88);
      goto LABEL_39;
    }
    v9 = "zeroinitializer";
    if ( v7 == 10 )
    {
LABEL_7:
      sub_1263B40(a1, v9);
      return;
    }
    if ( v7 == 4 )
    {
      sub_1263B40(a1, "blockaddress(");
      sub_1550E20(a1, *(v6 - 6), a3, a4, a5);
      sub_1263B40(a1, ", ");
      sub_1550E20(a1, *(v6 - 3), a3, a4, a5);
      v9 = ")";
      goto LABEL_7;
    }
    if ( v7 == 6 )
    {
      v89 = *(_QWORD *)(*v6 + 24);
      sub_1549FC0(a1, 0x5Bu);
      sub_154DAA0(a3, v89, a1);
      sub_1549FC0(a1, 0x20u);
      sub_1550E20(a1, v6[-3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)], a3, a4, a5);
      v83 = *((_DWORD *)v6 + 5) & 0xFFFFFFF;
      if ( v83 != 1 )
      {
        v79 = a4;
        for ( j = 1; j != v83; ++j )
        {
          sub_1263B40(a1, ", ");
          sub_154DAA0(a3, v89, a1);
          sub_1549FC0(a1, 0x20u);
          v27 = j;
          sub_1550E20(a1, v6[3 * (v27 - (*((_DWORD *)v6 + 5) & 0xFFFFFFF))], a3, v79, a5);
        }
      }
    }
    else
    {
      if ( v7 != 11 )
      {
        if ( v7 == 7 )
        {
          if ( (*(_BYTE *)(*v6 + 9) & 2) != 0 )
            sub_1549FC0(a1, 0x3Cu);
          sub_1549FC0(a1, 0x7Bu);
          v90 = *((_DWORD *)v6 + 5) & 0xFFFFFFF;
          if ( v90 )
          {
            sub_1549FC0(a1, 0x20u);
            sub_154DAA0(a3, *(_QWORD *)v6[-3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)], a1);
            sub_1549FC0(a1, 0x20u);
            sub_1550E20(a1, v6[-3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)], a3, a4, a5);
            if ( v90 != 1 )
            {
              v86 = a4;
              v54 = v6;
              v55 = a5;
              v56 = 1;
              do
              {
                sub_1263B40(a1, ", ");
                sub_154DAA0(a3, *(_QWORD *)v54[3 * (v56 - (*((_DWORD *)v54 + 5) & 0xFFFFFFF))], a1);
                sub_1549FC0(a1, 0x20u);
                v57 = v56++;
                sub_1550E20(a1, v54[3 * (v57 - (*((_DWORD *)v54 + 5) & 0xFFFFFFF))], a3, v86, v55);
              }
              while ( v90 > (unsigned int)v56 );
              v6 = v54;
            }
            sub_1549FC0(a1, 0x20u);
          }
          sub_1549FC0(a1, 0x7Du);
          if ( (*(_BYTE *)(*v6 + 9) & 2) == 0 )
            return;
        }
        else
        {
          if ( (v7 & 0xFB) != 8 )
          {
            if ( v7 == 15 )
            {
              v9 = "null";
              goto LABEL_7;
            }
            v9 = "none";
            if ( v7 == 16 )
              goto LABEL_7;
            v9 = "undef";
            if ( v7 == 9 )
              goto LABEL_7;
            if ( v7 != 5 )
            {
              v9 = "<placeholder or erroneous Constant>";
              goto LABEL_7;
            }
            v46 = (const char *)sub_1595080(v6, "undef");
            sub_1263B40(a1, v46);
            sub_154A7B0(a1, (__int64)v6);
            if ( (unsigned __int8)sub_1594520(v6, v6, v47, v48, v49) )
            {
              v94 = sub_1549FC0(a1, 0x20u);
              v62 = sub_1594720(v6);
              v63 = (const char *)sub_15FF290(v62);
              sub_1549FF0(v94, v63, v64);
            }
            sub_1263B40(a1, " (");
            v92 = 0;
            if ( *((_WORD *)v6 + 9) == 32 )
            {
              v50 = sub_16348C0(v6);
              sub_154DAA0(a3, v50, a1);
              sub_1263B40(a1, ", ");
              v77 = *((_BYTE *)v6 + 17) >> 1 >> 1;
              v92 = v77 != 0;
            }
            v51 = 24LL * (*((_DWORD *)v6 + 5) & 0xFFFFFFF);
            if ( &v6[v51 / 0xFFFFFFFFFFFFFFF8LL] != v6 )
            {
              v85 = a5;
              v52 = (__int64 **)&v6[v51 / 0xFFFFFFFFFFFFFFF8LL];
              if ( !v92 )
                goto LABEL_133;
LABEL_131:
              if ( v77 == -1431655765
                        * (unsigned int)(((char *)&v52[3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)] - (char *)v6) >> 3) )
                sub_1263B40(a1, "inrange ");
LABEL_133:
              while ( 1 )
              {
                v53 = *v52;
                v52 += 3;
                sub_154DAA0(a3, *v53, a1);
                sub_1549FC0(a1, 0x20u);
                sub_1550E20(a1, (__int64)*(v52 - 3), a3, a4, v85);
                if ( v52 == (__int64 **)v6 )
                  break;
                sub_1263B40(a1, ", ");
                if ( v92 )
                  goto LABEL_131;
              }
            }
            if ( (unsigned __int8)sub_1594700(v6) )
            {
              v66 = sub_1594710(v6);
              if ( v65 )
              {
                v67 = v66;
                v95 = v66 + 4LL * (unsigned int)(v65 - 1) + 4;
                do
                {
                  v67 += 4;
                  v68 = sub_1263B40(a1, ", ");
                  sub_16E7A90(v68, *(unsigned int *)(v67 - 4));
                }
                while ( v67 != v95 );
                v5 = a1;
              }
            }
            if ( (unsigned __int8)sub_1594510(v6) )
            {
              sub_1263B40(v5, " to ");
              sub_154DAA0(a3, *v6, v5);
            }
            v28 = 41;
LABEL_58:
            sub_1549FC0(v5, v28);
            return;
          }
          v93 = **(_QWORD **)(*v6 + 16);
          sub_1549FC0(a1, 0x3Cu);
          sub_154DAA0(a3, v93, a1);
          sub_1549FC0(a1, 0x20u);
          v58 = sub_15A0A60(v6, 0);
          sub_1550E20(a1, v58, a3, a4, a5);
          v87 = *(_QWORD *)(*v6 + 32);
          if ( v87 != 1 )
          {
            v81 = a5;
            for ( k = 1; k != v87; ++k )
            {
              sub_1263B40(a1, ", ");
              sub_154DAA0(a3, v93, a1);
              sub_1549FC0(a1, 0x20u);
              v60 = k;
              v61 = sub_15A0A60(v6, v60);
              sub_1550E20(a1, v61, a3, a4, v81);
            }
          }
        }
        v28 = 62;
        goto LABEL_58;
      }
      if ( (unsigned __int8)sub_1595C40(v6, 8) )
      {
        sub_1263B40(a1, "c\"");
        v30 = sub_1595920(v6);
        sub_16D16F0(v30, v31, a1);
        v28 = 34;
        goto LABEL_58;
      }
      v91 = *(_QWORD *)(*v6 + 24);
      sub_1549FC0(a1, 0x5Bu);
      sub_154DAA0(a3, v91, a1);
      sub_1549FC0(a1, 0x20u);
      v41 = sub_15A0940(v6, 0);
      sub_1550E20(a1, v41, a3, a4, a5);
      v84 = sub_15958F0(v6);
      if ( v84 != 1 )
      {
        v80 = a5;
        for ( m = 1; m != v84; ++m )
        {
          sub_1263B40(a1, ", ");
          sub_154DAA0(a3, v91, a1);
          sub_1549FC0(a1, 0x20u);
          v43 = m;
          v44 = sub_15A0940(v6, v43);
          sub_1550E20(a1, v44, a3, a4, v80);
        }
      }
    }
    v28 = 93;
    goto LABEL_58;
  }
  if ( (unsigned __int8)sub_1642F90(*a2, 1) )
  {
    v8 = (_QWORD *)a2[3];
    if ( *((_DWORD *)a2 + 8) > 0x40u )
      v8 = (_QWORD *)*v8;
    v9 = "true";
    if ( !v8 )
      v9 = "false";
    goto LABEL_7;
  }
  sub_16A95F0(a2 + 3, a1, 1);
}
