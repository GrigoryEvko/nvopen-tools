// Function: sub_1CB4F40
// Address: 0x1cb4f40
//
__int64 __fastcall sub_1CB4F40(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r12
  int v7; // ebx
  _QWORD *v8; // r13
  _QWORD *n; // r14
  __int64 v10; // rsi
  unsigned __int64 v11; // r13
  _QWORD *v12; // r14
  __int64 v13; // rsi
  _QWORD *v15; // r13
  _QWORD *i; // r11
  _QWORD *j; // rbx
  _QWORD *v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v26; // rax
  _QWORD *v27; // rcx
  char v28; // al
  const char *v29; // rax
  size_t v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rax
  _BYTE *v33; // rsi
  _QWORD *v34; // r11
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rax
  _BYTE *v39; // rsi
  _QWORD *v40; // r11
  _BYTE *v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // rax
  size_t v44; // rdx
  _QWORD *v45; // r11
  __int64 v46; // rcx
  char *v47; // rsi
  size_t v48; // rax
  int v49; // eax
  int v50; // ebx
  char *v51; // r14
  size_t v52; // rdx
  size_t v53; // r12
  __int64 *v54; // rax
  __int64 v55; // rax
  _BYTE *v56; // rsi
  _QWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // rax
  _BYTE *v61; // rsi
  _BYTE *v62; // rbx
  __int64 *v63; // r12
  _QWORD *v64; // r14
  __int64 *v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // r12
  _QWORD *m; // r15
  __int64 v69; // r11
  size_t v70; // rdx
  char *v71; // rsi
  size_t v72; // rax
  __int64 v73; // rax
  size_t v74; // rdx
  size_t v75; // rax
  __int64 v76; // [rsp+0h] [rbp-80h]
  _QWORD *v77; // [rsp+0h] [rbp-80h]
  _QWORD *v78; // [rsp+0h] [rbp-80h]
  __int64 v79; // [rsp+0h] [rbp-80h]
  char *v81; // [rsp+10h] [rbp-70h]
  __int64 v82; // [rsp+10h] [rbp-70h]
  __int64 v83; // [rsp+10h] [rbp-70h]
  __int64 v84; // [rsp+10h] [rbp-70h]
  _QWORD *v85; // [rsp+18h] [rbp-68h]
  _QWORD *k; // [rsp+18h] [rbp-68h]
  _QWORD *v87; // [rsp+18h] [rbp-68h]
  size_t v88; // [rsp+18h] [rbp-68h]
  _QWORD *v89; // [rsp+18h] [rbp-68h]
  __int64 *v90; // [rsp+18h] [rbp-68h]
  char *v91; // [rsp+18h] [rbp-68h]
  _QWORD *v92; // [rsp+18h] [rbp-68h]
  __int64 v93; // [rsp+18h] [rbp-68h]
  _QWORD *v94; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v95; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v96; // [rsp+38h] [rbp-48h]
  _BYTE *v97; // [rsp+40h] [rbp-40h]

  if ( byte_4FBEB60 )
  {
    v15 = (_QWORD *)a2[10];
    for ( i = a2 + 9; v15 != i; v15 = (_QWORD *)v15[1] )
    {
      if ( !v15 )
        BUG();
      for ( j = (_QWORD *)v15[3]; v15 + 2 != j; j = (_QWORD *)j[1] )
      {
        if ( !j )
          BUG();
        if ( *((_BYTE *)j - 8) == 56 )
        {
          v18 = j - 3;
          if ( *(_BYTE *)(*(_QWORD *)j[-3 * (*((_DWORD *)j - 1) & 0xFFFFFFF) - 3] + 8LL) == 15 )
          {
            v19 = j[4];
            if ( *(_BYTE *)(v19 + 8) == 13 && (*(_BYTE *)(v19 + 9) & 4) == 0 )
            {
              v87 = i;
              v20 = sub_1643640(v19);
              i = v87;
              v22 = v20;
              if ( v21 > 6 && *(_DWORD *)v20 == 1970435187 && *(_WORD *)(v20 + 4) == 29795 && *(_BYTE *)(v20 + 6) == 46 )
              {
                v21 -= 7LL;
                v22 = v20 + 7;
              }
              v23 = *((_DWORD *)j - 1) & 0xFFFFFFF;
              v24 = v18[3 * (1 - v23)];
              if ( *(_BYTE *)(v24 + 16) == 13
                && !(*(_DWORD *)(v24 + 32) <= 0x40u ? *(_QWORD *)(v24 + 24) : **(_QWORD **)(v24 + 24))
                && (*((_DWORD *)j - 1) & 0xFFFFFFFu) > 2 )
              {
                v26 = v18[3 * (2 - v23)];
                if ( *(_BYTE *)(v26 + 16) == 13 )
                {
                  v27 = *(_QWORD **)(v26 + 24);
                  if ( *(_DWORD *)(v26 + 32) > 0x40u )
                    v27 = (_QWORD *)*v27;
                  v28 = sub_1CCACD0(a2[5], v22, v21, v27);
                  i = v87;
                  if ( v28 )
                  {
                    v77 = v87;
                    v95 = 0;
                    v96 = 0;
                    v97 = 0;
                    v29 = sub_1649960((__int64)a2);
                    v88 = v30;
                    v81 = (char *)v29;
                    v31 = (__int64 *)sub_16498A0((__int64)(j - 3));
                    v32 = sub_161FF10(v31, v81, v88);
                    v33 = v96;
                    v94 = (_QWORD *)v32;
                    v34 = v77;
                    if ( v96 == v97 )
                    {
                      sub_1273E00((__int64)&v95, v96, &v94);
                      v34 = v77;
                    }
                    else
                    {
                      if ( v96 )
                      {
                        *(_QWORD *)v96 = v32;
                        v33 = v96;
                      }
                      v96 = v33 + 8;
                    }
                    v89 = v34;
                    v35 = (_QWORD *)sub_16498A0((__int64)(j - 3));
                    v36 = sub_1643350(v35);
                    v37 = sub_159C470(v36, 0, 0);
                    v38 = sub_1624210(v37);
                    v39 = v96;
                    v94 = v38;
                    v40 = v89;
                    if ( v96 == v97 )
                    {
                      sub_1273E00((__int64)&v95, v96, &v94);
                      v41 = v96;
                      v40 = v89;
                    }
                    else
                    {
                      if ( v96 )
                      {
                        *(_QWORD *)v96 = v38;
                        v39 = v96;
                      }
                      v41 = v39 + 8;
                      v96 = v41;
                    }
                    v78 = v40;
                    v90 = v95;
                    v42 = (__int64 *)sub_16498A0((__int64)(j - 3));
                    v43 = sub_1627350(v42, v90, (__int64 *)((v41 - (_BYTE *)v90) >> 3), 0, 1);
                    v44 = 0;
                    v45 = v78;
                    v46 = v43;
                    v47 = off_4CD4978[0];
                    if ( off_4CD4978[0] )
                    {
                      v82 = v43;
                      v91 = off_4CD4978[0];
                      v48 = strlen(off_4CD4978[0]);
                      v45 = v78;
                      v46 = v82;
                      v47 = v91;
                      v44 = v48;
                    }
                    v92 = v45;
                    sub_1626100((__int64)(j - 3), v47, v44, v46);
                    sub_1CCAB50(1, a2);
                    sub_1CCABF0(3, a2);
                    sub_1CCABF0(2, a2);
                    i = v92;
                    if ( v95 )
                    {
                      j_j___libc_free_0(v95, v97 - (_BYTE *)v95);
                      i = v92;
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
  v3 = sub_1CCAAC0(1, a2);
  if ( (_BYTE)v3 )
  {
    v4 = a2[10];
    v5 = a2[9];
    v76 = v4;
    if ( v4 )
    {
      v6 = v4 - 24;
      v85 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v4 == (v5 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v7 = 0;
LABEL_15:
        v12 = (_QWORD *)(v85[2] & 0xFFFFFFFFFFFFFFF8LL);
        for ( k = (_QWORD *)v85[3]; k != v12; v12 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v13 = (__int64)(v12 - 3);
          if ( !v12 )
            v13 = 0;
          v7 |= sub_1CB4290(a1, v13, v6, *(_BYTE *)(a1 + 16));
        }
        if ( v12 )
          v12 -= 3;
        v3 = v7 | sub_1CB4290(a1, (__int64)v12, v6, *(_BYTE *)(a1 + 16));
        if ( (unsigned __int8)sub_1CCAAC0(2, a2) )
          goto LABEL_22;
        v49 = sub_1CCB210();
        v95 = 0;
        v96 = 0;
        v50 = v49;
        v97 = 0;
        v51 = (char *)sub_1649960((__int64)a2);
        v53 = v52;
        v54 = (__int64 *)sub_15E0530((__int64)a2);
        v55 = sub_161FF10(v54, v51, v53);
        v56 = v96;
        v94 = (_QWORD *)v55;
        if ( v96 == v97 )
        {
          sub_1273E00((__int64)&v95, v96, &v94);
        }
        else
        {
          if ( v96 )
          {
            *(_QWORD *)v96 = v55;
            v56 = v96;
          }
          v96 = v56 + 8;
        }
        v57 = (_QWORD *)sub_15E0530((__int64)a2);
        v58 = sub_1643350(v57);
        v59 = sub_159C470(v58, v50, 0);
        v60 = sub_1624210(v59);
        v61 = v96;
        v94 = v60;
        if ( v96 == v97 )
        {
          sub_1273E00((__int64)&v95, v96, &v94);
          v62 = v96;
        }
        else
        {
          if ( v96 )
          {
            *(_QWORD *)v96 = v60;
            v61 = v96;
          }
          v62 = v61 + 8;
          v96 = v61 + 8;
        }
        v63 = v95;
        v64 = a2 + 9;
        v65 = (__int64 *)sub_15E0530((__int64)a2);
        v66 = sub_1627350(v65, v63, (__int64 *)((v62 - (_BYTE *)v63) >> 3), 0, 1);
        v67 = (_QWORD *)a2[10];
        v79 = v66;
        if ( v67 == a2 + 9 )
          goto LABEL_72;
        if ( !v67 )
          BUG();
        while ( 1 )
        {
          m = (_QWORD *)v67[3];
          if ( m != v67 + 2 )
            break;
          v67 = (_QWORD *)v67[1];
          if ( v67 == v64 )
            goto LABEL_72;
          if ( !v67 )
            BUG();
        }
        if ( v67 == v64 )
        {
LABEL_72:
          sub_1CCAB50(2, a2);
          if ( v95 )
            j_j___libc_free_0(v95, v97 - (_BYTE *)v95);
          goto LABEL_22;
        }
        while ( 1 )
        {
          v69 = (__int64)(m - 3);
          if ( !m )
            v69 = 0;
          v70 = 0;
          v71 = off_4CD4970[0];
          if ( off_4CD4970[0] )
          {
            v83 = v69;
            v72 = strlen(off_4CD4970[0]);
            v69 = v83;
            v70 = v72;
          }
          if ( *(_QWORD *)(v69 + 48) || *(__int16 *)(v69 + 18) < 0 )
          {
            v93 = v69;
            v73 = sub_1625940(v69, v71, v70);
            v69 = v93;
            if ( v73 )
              goto LABEL_87;
            v71 = off_4CD4970[0];
          }
          v74 = 0;
          if ( v71 )
          {
            v84 = v69;
            v75 = strlen(v71);
            v69 = v84;
            v74 = v75;
          }
          sub_1626100(v69, v71, v74, v79);
LABEL_87:
          for ( m = (_QWORD *)m[1]; m == v67 + 2; m = (_QWORD *)v67[3] )
          {
            v67 = (_QWORD *)v67[1];
            if ( v67 == v64 )
              goto LABEL_72;
            if ( !v67 )
              BUG();
          }
          if ( v67 == v64 )
            goto LABEL_72;
        }
      }
    }
    else
    {
      v85 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_108;
      v6 = 0;
    }
    v7 = 0;
    do
    {
      if ( !v85 )
        BUG();
      v8 = (_QWORD *)v85[3];
      for ( n = (_QWORD *)(v85[2] & 0xFFFFFFFFFFFFFFF8LL); v8 != n; n = (_QWORD *)(*n & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v10 = (__int64)(n - 3);
        if ( !n )
          v10 = 0;
        v7 |= sub_1CB4290(a1, v10, v6, *(_BYTE *)(a1 + 16));
      }
      if ( n )
        n -= 3;
      v7 |= sub_1CB4290(a1, (__int64)n, v6, *(_BYTE *)(a1 + 16));
      v11 = *v85 & 0xFFFFFFFFFFFFFFF8LL;
      v85 = (_QWORD *)v11;
    }
    while ( v76 != v11 );
    if ( v11 )
      goto LABEL_15;
LABEL_108:
    BUG();
  }
LABEL_22:
  sub_1CCAB50(3, a2);
  return v3;
}
