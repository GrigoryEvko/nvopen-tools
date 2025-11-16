// Function: sub_2CFE410
// Address: 0x2cfe410
//
__int64 __fastcall sub_2CFE410(__int64 a1, __int64 *a2)
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
  __int64 v15; // r14
  __int64 *i; // r13
  __int64 j; // rbx
  __int64 v18; // rdi
  __int64 v19; // rax
  size_t v20; // rdx
  const void *v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rcx
  _QWORD *v24; // r8
  __int64 v25; // rax
  _QWORD *v26; // rcx
  const char *v27; // rax
  size_t v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // rsi
  __int64 v32; // r10
  _QWORD *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rax
  _BYTE *v37; // rsi
  __int64 v38; // r10
  _BYTE *v39; // rsi
  __int64 *v40; // rax
  __int64 v41; // rax
  size_t v42; // rdx
  __int64 v43; // r10
  __int64 v44; // rcx
  char *v45; // rsi
  size_t v46; // rax
  int v47; // eax
  int v48; // ebx
  const char *v49; // r14
  size_t v50; // rdx
  size_t v51; // r12
  __int64 *v52; // rax
  __int64 v53; // rax
  _BYTE *v54; // rsi
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  _QWORD *v58; // rax
  _BYTE *v59; // rsi
  _BYTE *v60; // r14
  __int64 *v61; // r12
  __int64 *v62; // rbx
  __int64 *v63; // rax
  __int64 v64; // rax
  __int64 *v65; // r12
  __int64 *m; // r14
  __int64 v67; // r11
  size_t v68; // rdx
  char *v69; // rsi
  size_t v70; // rax
  __int64 v71; // rax
  size_t v72; // rdx
  size_t v73; // rax
  __int64 v74; // [rsp+0h] [rbp-80h]
  const char *v75; // [rsp+0h] [rbp-80h]
  __int64 v76; // [rsp+0h] [rbp-80h]
  __int64 v77; // [rsp+0h] [rbp-80h]
  __int64 *v79; // [rsp+10h] [rbp-70h]
  __int64 v80; // [rsp+10h] [rbp-70h]
  __int64 v81; // [rsp+10h] [rbp-70h]
  __int64 v82; // [rsp+10h] [rbp-70h]
  _QWORD *v83; // [rsp+18h] [rbp-68h]
  _QWORD *k; // [rsp+18h] [rbp-68h]
  __int64 v85; // [rsp+18h] [rbp-68h]
  size_t v86; // [rsp+18h] [rbp-68h]
  __int64 v87; // [rsp+18h] [rbp-68h]
  __int64 v88; // [rsp+18h] [rbp-68h]
  char *v89; // [rsp+18h] [rbp-68h]
  __int64 v90; // [rsp+18h] [rbp-68h]
  _QWORD *v91; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v92; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v93; // [rsp+38h] [rbp-48h]
  _BYTE *v94; // [rsp+40h] [rbp-40h]

  if ( (_BYTE)qword_5014C08 )
  {
    v15 = a2[10];
    for ( i = a2 + 9; (__int64 *)v15 != i; v15 = *(_QWORD *)(v15 + 8) )
    {
      if ( !v15 )
        BUG();
      for ( j = *(_QWORD *)(v15 + 32); j != v15 + 24; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          BUG();
        if ( *(_BYTE *)(j - 24) == 63
          && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(j - 32LL * (*(_DWORD *)(j - 20) & 0x7FFFFFF) - 24) + 8LL) + 8LL) == 14 )
        {
          v18 = *(_QWORD *)(j + 48);
          if ( *(_BYTE *)(v18 + 8) == 15 && (*(_BYTE *)(v18 + 9) & 4) == 0 )
          {
            v85 = j - 24;
            v19 = sub_BCB490(v18);
            v21 = (const void *)v19;
            if ( v20 > 6 && *(_DWORD *)v19 == 1970435187 && *(_WORD *)(v19 + 4) == 29795 && *(_BYTE *)(v19 + 6) == 46 )
            {
              v20 -= 7LL;
              v21 = (const void *)(v19 + 7);
            }
            v22 = *(_DWORD *)(j - 20) & 0x7FFFFFF;
            v23 = *(_QWORD *)(v85 + 32 * (1 - v22));
            if ( *(_BYTE *)v23 == 17 )
            {
              v24 = *(_QWORD **)(v23 + 24);
              if ( *(_DWORD *)(v23 + 32) > 0x40u )
                v24 = (_QWORD *)*v24;
              if ( (*(_DWORD *)(j - 20) & 0x7FFFFFFu) > 2 && !v24 )
              {
                v25 = *(_QWORD *)(v85 + 32 * (2 - v22));
                if ( *(_BYTE *)v25 == 17 )
                {
                  v26 = *(_QWORD **)(v25 + 24);
                  if ( *(_DWORD *)(v25 + 32) > 0x40u )
                    v26 = (_QWORD *)*v26;
                  if ( (unsigned __int8)sub_CEF9E0(a2[5], v21, v20, (unsigned int)v26) )
                  {
                    v92 = 0;
                    v93 = 0;
                    v94 = 0;
                    v27 = sub_BD5D20((__int64)a2);
                    v86 = v28;
                    v75 = v27;
                    v29 = (__int64 *)sub_BD5C60(j - 24);
                    v30 = sub_B9B140(v29, v75, v86);
                    v31 = v93;
                    v91 = (_QWORD *)v30;
                    v32 = j - 24;
                    if ( v93 == v94 )
                    {
                      sub_914280((__int64)&v92, v93, &v91);
                      v32 = j - 24;
                    }
                    else
                    {
                      if ( v93 )
                      {
                        *(_QWORD *)v93 = v30;
                        v31 = v93;
                      }
                      v93 = v31 + 8;
                    }
                    v87 = v32;
                    v33 = (_QWORD *)sub_BD5C60(v32);
                    v34 = sub_BCB2D0(v33);
                    v35 = sub_ACD640(v34, 0, 0);
                    v36 = sub_B98A20(v35, 0);
                    v37 = v93;
                    v91 = v36;
                    v38 = v87;
                    if ( v93 == v94 )
                    {
                      sub_914280((__int64)&v92, v93, &v91);
                      v39 = v93;
                      v38 = v87;
                    }
                    else
                    {
                      if ( v93 )
                      {
                        *(_QWORD *)v93 = v36;
                        v37 = v93;
                      }
                      v39 = v37 + 8;
                      v93 = v39;
                    }
                    v88 = v38;
                    v79 = v92;
                    v40 = (__int64 *)sub_BD5C60(v38);
                    v41 = sub_B9C770(v40, v79, (__int64 *)((v39 - (_BYTE *)v79) >> 3), 0, 1);
                    v42 = 0;
                    v43 = v88;
                    v44 = v41;
                    v45 = off_4C5D0D8[0];
                    if ( off_4C5D0D8[0] )
                    {
                      v76 = v88;
                      v80 = v41;
                      v89 = off_4C5D0D8[0];
                      v46 = strlen(off_4C5D0D8[0]);
                      v43 = v76;
                      v44 = v80;
                      v45 = v89;
                      v42 = v46;
                    }
                    sub_B9A090(v43, v45, v42, v44);
                    sub_CEF870(1, (__int64)a2);
                    sub_CEF900(3, (__int64)a2);
                    sub_CEF900(2, (__int64)a2);
                    if ( v92 )
                      j_j___libc_free_0((unsigned __int64)v92);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v3 = sub_CEF7D0(1, (__int64)a2);
  if ( (_BYTE)v3 )
  {
    v4 = a2[10];
    v5 = a2[9];
    v74 = v4;
    if ( v4 )
    {
      v6 = v4 - 24;
      v83 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v4 == (v5 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v7 = 0;
LABEL_15:
        v12 = (_QWORD *)(v83[3] & 0xFFFFFFFFFFFFFFF8LL);
        for ( k = (_QWORD *)v83[4]; k != v12; v12 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v13 = (__int64)(v12 - 3);
          if ( !v12 )
            v13 = 0;
          v7 |= sub_2CFD5A0(a1, v13, v6, *(_BYTE *)(a1 + 16));
        }
        if ( v12 )
          v12 -= 3;
        v3 = v7 | sub_2CFD5A0(a1, (__int64)v12, v6, *(_BYTE *)(a1 + 16));
        if ( (unsigned __int8)sub_CEF7D0(2, (__int64)a2) )
          goto LABEL_22;
        v47 = sub_CEFE00();
        v92 = 0;
        v93 = 0;
        v48 = v47;
        v94 = 0;
        v49 = sub_BD5D20((__int64)a2);
        v51 = v50;
        v52 = (__int64 *)sub_B2BE50((__int64)a2);
        v53 = sub_B9B140(v52, v49, v51);
        v54 = v93;
        v91 = (_QWORD *)v53;
        if ( v93 == v94 )
        {
          sub_914280((__int64)&v92, v93, &v91);
        }
        else
        {
          if ( v93 )
          {
            *(_QWORD *)v93 = v53;
            v54 = v93;
          }
          v93 = v54 + 8;
        }
        v55 = (_QWORD *)sub_B2BE50((__int64)a2);
        v56 = sub_BCB2D0(v55);
        v57 = sub_ACD640(v56, v48, 0);
        v58 = sub_B98A20(v57, v48);
        v59 = v93;
        v91 = v58;
        if ( v93 == v94 )
        {
          sub_914280((__int64)&v92, v93, &v91);
          v60 = v93;
        }
        else
        {
          if ( v93 )
          {
            *(_QWORD *)v93 = v58;
            v59 = v93;
          }
          v60 = v59 + 8;
          v93 = v59 + 8;
        }
        v61 = v92;
        v62 = a2 + 9;
        v63 = (__int64 *)sub_B2BE50((__int64)a2);
        v64 = sub_B9C770(v63, v61, (__int64 *)((v60 - (_BYTE *)v61) >> 3), 0, 1);
        v65 = (__int64 *)a2[10];
        v77 = v64;
        if ( a2 + 9 == v65 )
          goto LABEL_72;
        if ( !v65 )
          BUG();
        while ( 1 )
        {
          m = (__int64 *)v65[4];
          if ( m != v65 + 3 )
            break;
          v65 = (__int64 *)v65[1];
          if ( v62 == v65 )
            goto LABEL_72;
          if ( !v65 )
            BUG();
        }
        if ( v62 == v65 )
        {
LABEL_72:
          sub_CEF870(2, (__int64)a2);
          if ( v92 )
            j_j___libc_free_0((unsigned __int64)v92);
          goto LABEL_22;
        }
        while ( 1 )
        {
          v67 = (__int64)(m - 3);
          if ( !m )
            v67 = 0;
          v68 = 0;
          v69 = off_4C5D0D0[0];
          if ( off_4C5D0D0[0] )
          {
            v81 = v67;
            v70 = strlen(off_4C5D0D0[0]);
            v67 = v81;
            v68 = v70;
          }
          if ( *(_QWORD *)(v67 + 48) || (*(_BYTE *)(v67 + 7) & 0x20) != 0 )
          {
            v90 = v67;
            v71 = sub_B91F50(v67, v69, v68);
            v67 = v90;
            if ( v71 )
              goto LABEL_86;
            v69 = off_4C5D0D0[0];
          }
          v72 = 0;
          if ( v69 )
          {
            v82 = v67;
            v73 = strlen(v69);
            v67 = v82;
            v72 = v73;
          }
          sub_B9A090(v67, v69, v72, v77);
LABEL_86:
          for ( m = (__int64 *)m[1]; m == v65 + 3; m = (__int64 *)v65[4] )
          {
            v65 = (__int64 *)v65[1];
            if ( v62 == v65 )
              goto LABEL_72;
            if ( !v65 )
              BUG();
          }
          if ( v62 == v65 )
            goto LABEL_72;
        }
      }
    }
    else
    {
      v83 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_107;
      v6 = 0;
    }
    v7 = 0;
    do
    {
      if ( !v83 )
        BUG();
      v8 = (_QWORD *)v83[4];
      for ( n = (_QWORD *)(v83[3] & 0xFFFFFFFFFFFFFFF8LL); v8 != n; n = (_QWORD *)(*n & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v10 = (__int64)(n - 3);
        if ( !n )
          v10 = 0;
        v7 |= sub_2CFD5A0(a1, v10, v6, *(_BYTE *)(a1 + 16));
      }
      if ( n )
        n -= 3;
      v7 |= sub_2CFD5A0(a1, (__int64)n, v6, *(_BYTE *)(a1 + 16));
      v11 = *v83 & 0xFFFFFFFFFFFFFFF8LL;
      v83 = (_QWORD *)v11;
    }
    while ( v74 != v11 );
    if ( v11 )
      goto LABEL_15;
LABEL_107:
    BUG();
  }
LABEL_22:
  sub_CEF870(3, (__int64)a2);
  return v3;
}
