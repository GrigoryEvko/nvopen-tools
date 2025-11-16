// Function: sub_32379A0
// Address: 0x32379a0
//
char __fastcall sub_32379A0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  char result; // al
  unsigned __int8 v11; // al
  bool v12; // dl
  size_t v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  size_t v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r8
  size_t v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  bool v23; // zf
  unsigned __int8 v24; // dl
  __int64 v25; // r8
  size_t v26; // rdx
  size_t v27; // r9
  size_t v28; // rcx
  __int64 v29; // rdi
  const void *v30; // rax
  __int64 v31; // rdx
  int v32; // r8d
  int v33; // esi
  __int64 v34; // rdi
  int v35; // esi
  unsigned int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // r8
  size_t v39; // rdx
  _BYTE *v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rsi
  unsigned __int8 v43; // al
  size_t v44; // rdx
  _BYTE *v45; // r10
  __int64 v46; // rdx
  __int64 v47; // r11
  char *v48; // r10
  size_t v49; // r11
  _BYTE *v50; // rax
  _BYTE *v51; // rax
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // rax
  __int64 v54; // r9
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rdx
  size_t v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // r8
  __int64 v62; // rax
  size_t v63; // rdx
  char *v64; // rdi
  _BYTE *v65; // rax
  _BYTE *v66; // rax
  char *v67; // r10
  size_t v68; // r11
  unsigned __int64 v69; // r8
  unsigned __int64 v70; // rax
  char *v71; // rcx
  _BYTE *v72; // rax
  _BYTE *v73; // rax
  unsigned __int64 v74; // r9
  __int64 v75; // rcx
  __int64 v76; // r8
  unsigned __int64 v77; // rax
  unsigned __int8 v78; // al
  size_t v79; // rdx
  _BYTE *v80; // rbx
  _BYTE *v81; // rax
  size_t v82; // rdx
  _BYTE *v83; // rax
  _BYTE *v84; // rax
  size_t v85; // rdx
  unsigned __int64 v86; // r8
  unsigned __int64 v87; // rax
  _BYTE *v88; // rax
  _BYTE *v89; // rax
  _BYTE *v90; // rax
  _BYTE *v91; // rax
  int v92; // edx
  int v93; // r9d
  __int64 v94; // [rsp+0h] [rbp-70h]
  unsigned __int64 v95; // [rsp+8h] [rbp-68h]
  __int64 v96; // [rsp+8h] [rbp-68h]
  unsigned __int64 v97; // [rsp+10h] [rbp-60h]
  unsigned __int64 v98; // [rsp+10h] [rbp-60h]
  __int64 v99; // [rsp+10h] [rbp-60h]
  unsigned __int64 v100; // [rsp+10h] [rbp-60h]
  void *s2; // [rsp+18h] [rbp-58h]
  void *s2a; // [rsp+18h] [rbp-58h]
  char *s2b; // [rsp+18h] [rbp-58h]
  void *s2c; // [rsp+18h] [rbp-58h]
  void *s2d; // [rsp+18h] [rbp-58h]
  void *s2e; // [rsp+18h] [rbp-58h]
  bool n; // [rsp+20h] [rbp-50h]
  size_t na; // [rsp+20h] [rbp-50h]
  _BYTE *nb; // [rsp+20h] [rbp-50h]
  size_t nc; // [rsp+20h] [rbp-50h]
  _BYTE *nd; // [rsp+20h] [rbp-50h]
  char *ne; // [rsp+20h] [rbp-50h]
  size_t nf; // [rsp+20h] [rbp-50h]
  size_t ng; // [rsp+20h] [rbp-50h]
  size_t v115; // [rsp+28h] [rbp-48h]
  size_t v116; // [rsp+28h] [rbp-48h]
  __int64 v117[8]; // [rsp+30h] [rbp-40h] BYREF

  result = a3 == 2;
  n = a3 == 2 && *(_DWORD *)(a1 + 3764) != 2;
  if ( n || (*(_BYTE *)(a4 + 36) & 8) == 0 )
    return result;
  v115 = a4 - 16;
  v11 = *(_BYTE *)(a4 - 16);
  v12 = (v11 & 2) != 0;
  if ( (v11 & 2) != 0 )
    v13 = *(_QWORD *)(a4 - 32);
  else
    v13 = v115 - 8LL * ((v11 >> 2) & 0xF);
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 )
  {
    sub_B91420(v14);
    v11 = *(_BYTE *)(a4 - 16);
    if ( v15 )
    {
      if ( (v11 & 2) != 0 )
        v16 = *(_QWORD *)(a4 - 32);
      else
        v16 = v115 - 8LL * ((v11 >> 2) & 0xF);
      v17 = *(_QWORD *)(v16 + 16);
      if ( v17 )
      {
        v17 = sub_B91420(*(_QWORD *)(v16 + 16));
        v19 = v18;
      }
      else
      {
        v19 = 0;
      }
      sub_3237930(a1, a2, a3, v17, v19, a5);
      v11 = *(_BYTE *)(a4 - 16);
    }
    v12 = (v11 & 2) != 0;
  }
  if ( v12 )
    v20 = *(_QWORD *)(a4 - 32);
  else
    v20 = v115 - 8LL * ((v11 >> 2) & 0xF);
  v21 = *(_QWORD *)(v20 + 24);
  if ( v21 )
  {
    sub_B91420(v21);
    v11 = *(_BYTE *)(a4 - 16);
    v23 = v22 == 0;
    v24 = v11;
    if ( v23 )
    {
LABEL_64:
      v12 = (v24 & 2) != 0;
      goto LABEL_32;
    }
    if ( (v11 & 2) != 0 )
    {
      v55 = *(_QWORD *)(a4 - 32);
      v25 = *(_QWORD *)(v55 + 24);
      if ( !v25 )
      {
        v56 = *(_QWORD *)(v55 + 16);
        if ( !v56 )
        {
          n = (v11 & 2) != 0;
          goto LABEL_31;
        }
        sub_B91420(v56);
        if ( !v57 )
          goto LABEL_149;
        goto LABEL_57;
      }
    }
    else
    {
      v25 = *(_QWORD *)(v115 - 8LL * ((v11 >> 2) & 0xF) + 24);
      if ( !v25 )
      {
        v27 = 0;
        goto LABEL_67;
      }
    }
    v25 = sub_B91420(v25);
    v11 = *(_BYTE *)(a4 - 16);
    v27 = v26;
    if ( (v11 & 2) != 0 )
    {
      n = 1;
      v28 = *(_QWORD *)(a4 - 32);
      goto LABEL_22;
    }
LABEL_67:
    v28 = v115 - 8LL * ((v11 >> 2) & 0xF);
LABEL_22:
    v29 = *(_QWORD *)(v28 + 16);
    s2 = (void *)v25;
    if ( !v29 )
    {
      if ( !v27 )
        goto LABEL_31;
      goto LABEL_27;
    }
    na = v27;
    v30 = (const void *)sub_B91420(v29);
    if ( na == v31 )
    {
      if ( na )
      {
        v32 = memcmp(v30, s2, na);
        v11 = *(_BYTE *)(a4 - 16);
        v12 = (v11 & 2) != 0;
        if ( !v32 )
          goto LABEL_32;
        n = (*(_BYTE *)(a4 - 16) & 2) != 0;
LABEL_27:
        if ( !*(_BYTE *)(a1 + 3686) )
        {
          v33 = *(_DWORD *)(a1 + 3504);
          v34 = *(_QWORD *)(a1 + 3488);
          if ( !v33 )
            goto LABEL_31;
          v35 = v33 - 1;
          v36 = v35 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
          v37 = (__int64 *)(v34 + 16LL * v36);
          v38 = *v37;
          if ( a4 != *v37 )
          {
            v92 = 1;
            while ( v38 != -4096 )
            {
              v93 = v92 + 1;
              v36 = v35 & (v92 + v36);
              v37 = (__int64 *)(v34 + 16LL * v36);
              v38 = *v37;
              if ( a4 == *v37 )
                goto LABEL_30;
              v92 = v93;
            }
            goto LABEL_31;
          }
LABEL_30:
          if ( !v37[1] )
          {
LABEL_31:
            v12 = n;
            goto LABEL_32;
          }
        }
        if ( n )
          v58 = *(_QWORD *)(a4 - 32);
        else
          v58 = v115 - 8LL * ((v11 >> 2) & 0xF);
        v59 = *(_QWORD *)(v58 + 24);
        if ( v59 )
        {
          v59 = sub_B91420(*(_QWORD *)(v58 + 24));
          v61 = v60;
        }
        else
        {
          v61 = 0;
        }
        sub_3237930(a1, a2, a3, v59, v61, a5);
        v11 = *(_BYTE *)(a4 - 16);
        v24 = v11;
        goto LABEL_64;
      }
LABEL_149:
      v11 = *(_BYTE *)(a4 - 16);
      n = (v11 & 2) != 0;
      goto LABEL_31;
    }
LABEL_57:
    v11 = *(_BYTE *)(a4 - 16);
    n = (v11 & 2) != 0;
    goto LABEL_27;
  }
LABEL_32:
  if ( v12 )
    v39 = *(_QWORD *)(a4 - 32);
  else
    v39 = v115 - 8LL * ((v11 >> 2) & 0xF);
  v40 = *(_BYTE **)(v39 + 16);
  if ( v40 )
  {
    v40 = (_BYTE *)sub_B91420((__int64)v40);
    v42 = v41;
  }
  else
  {
    v42 = 0;
  }
  result = sub_32192A0(v40, v42);
  if ( result )
  {
    v43 = *(_BYTE *)(a4 - 16);
    if ( (v43 & 2) != 0 )
      v44 = *(_QWORD *)(a4 - 32);
    else
      v44 = v115 - 8LL * ((v43 >> 2) & 0xF);
    v45 = *(_BYTE **)(v44 + 16);
    if ( v45 )
    {
      v45 = (_BYTE *)sub_B91420(*(_QWORD *)(v44 + 16));
      v47 = v46;
    }
    else
    {
      v47 = 0;
    }
    v117[0] = (__int64)v45;
    v117[1] = v47;
    if ( sub_32192A0(v45, v47) )
    {
      s2b = v48;
      nc = v49;
      v62 = sub_C931B0(v117, ") ", 2u, 0);
      v49 = nc;
      v48 = s2b;
      if ( v62 != -1 )
      {
        if ( !nc )
        {
          v75 = (__int64)s2b;
          v76 = 0;
LABEL_103:
          s2e = (void *)v49;
          ng = (size_t)v48;
          sub_3237960(a1, a2, a3, v75, v76, a5);
          v49 = (size_t)s2e;
          if ( !s2e )
          {
LABEL_82:
            v78 = *(_BYTE *)(a4 - 16);
            if ( (v78 & 2) != 0 )
              v79 = *(_QWORD *)(a4 - 32);
            else
              v79 = v115 - 8LL * ((v78 >> 2) & 0xF);
            v80 = *(_BYTE **)(v79 + 16);
            if ( !v80 || (v81 = (_BYTE *)sub_B91420(*(_QWORD *)(v79 + 16)), v80 = v81, !v82) )
            {
LABEL_91:
              v86 = 0;
              return sub_3237930(a1, a2, a3, (__int64)v80, v86, a5);
            }
            v116 = v82;
            v83 = memchr(v81, 93, v82);
            if ( v83 )
            {
              nf = v83 - v80;
              v84 = memchr(v80, 32, v116);
              v85 = v116;
              v86 = nf;
              if ( v84 )
              {
                v87 = v84 - v80 + 1;
                if ( v87 > v116 )
                  v87 = v116;
                v80 += v87;
                if ( nf < v87 )
                  goto LABEL_91;
LABEL_114:
                if ( v85 <= v86 )
                  v86 = v85;
                v86 -= v87;
                return sub_3237930(a1, a2, a3, (__int64)v80, v86, a5);
              }
            }
            else
            {
              v89 = memchr(v80, 32, v116);
              v85 = v116;
              v86 = -1;
              if ( v89 )
              {
                v86 = -1;
                v87 = v89 - v80 + 1;
                if ( v87 > v116 )
                  v87 = v116;
                v80 += v87;
                goto LABEL_114;
              }
            }
            if ( v86 > v85 )
              v86 = v85;
            return sub_3237930(a1, a2, a3, (__int64)v80, v86, a5);
          }
          v48 = (char *)ng;
          v54 = a5;
LABEL_50:
          sub_3237960(a1, a2, a3, (__int64)v48, v49, v54);
          goto LABEL_82;
        }
        v63 = nc;
        v64 = s2b;
        s2c = (void *)nc;
        nd = v48;
        v65 = memchr(v64, 40, v63);
        if ( v65 )
        {
          v98 = v65 - nd;
          v66 = memchr(nd, 91, (size_t)s2c);
          v67 = nd;
          v68 = (size_t)s2c;
          v69 = v98;
          if ( v66 )
          {
            v70 = v66 - nd + 1;
            if ( v70 > (unsigned __int64)s2c )
              v70 = (unsigned __int64)s2c;
            v71 = &nd[v70];
            if ( v98 < v70 )
            {
              v69 = 0;
              goto LABEL_76;
            }
            goto LABEL_145;
          }
        }
        else
        {
          v88 = memchr(nd, 91, (size_t)s2c);
          v67 = nd;
          v68 = (size_t)s2c;
          v69 = -1;
          if ( v88 )
          {
            v69 = -1;
            v70 = v88 - nd + 1;
            if ( v70 > (unsigned __int64)s2c )
              v70 = (unsigned __int64)s2c;
            v71 = &nd[v70];
LABEL_145:
            if ( v68 <= v69 )
              v69 = v68;
            v69 -= v70;
            goto LABEL_76;
          }
        }
        v71 = v67;
        if ( v69 > v68 )
          v69 = v68;
LABEL_76:
        v95 = v69;
        v99 = (__int64)v71;
        s2d = (void *)v68;
        ne = v67;
        v72 = memchr(v67, 32, v68);
        if ( v72 )
        {
          v94 = v95;
          v96 = v99;
          v100 = v72 - ne;
          v73 = memchr(ne, 91, (size_t)s2d);
          v48 = ne;
          v49 = (size_t)s2d;
          v74 = v100;
          v75 = v96;
          v76 = v94;
          if ( v73 )
          {
            v77 = v73 - ne + 1;
            if ( v77 > (unsigned __int64)s2d )
              v77 = (unsigned __int64)s2d;
            v48 = &ne[v77];
            if ( v100 < v77 )
            {
              sub_3237960(a1, a2, a3, v96, v94, a5);
              goto LABEL_82;
            }
            goto LABEL_124;
          }
        }
        else
        {
          v90 = memchr(ne, 91, (size_t)s2d);
          v48 = ne;
          v49 = (size_t)s2d;
          v74 = -1;
          v75 = v99;
          v76 = v95;
          if ( v90 )
          {
            v74 = -1;
            v77 = v90 - ne + 1;
            if ( v77 > (unsigned __int64)s2d )
              v77 = (unsigned __int64)s2d;
            v48 = &ne[v77];
LABEL_124:
            if ( v74 <= v49 )
              v49 = v74;
            v49 -= v77;
            goto LABEL_103;
          }
        }
        if ( v49 > v74 )
          v49 = v74;
        goto LABEL_103;
      }
    }
    if ( !v49 )
    {
LABEL_49:
      v54 = a5;
      goto LABEL_50;
    }
    s2a = (void *)v49;
    nb = v48;
    v50 = memchr(v48, 32, v49);
    if ( v50 )
    {
      v97 = v50 - nb;
      v51 = memchr(nb, 91, (size_t)s2a);
      v48 = nb;
      v49 = (size_t)s2a;
      v52 = v97;
      if ( v51 )
      {
        v53 = v51 - nb + 1;
        if ( v53 > (unsigned __int64)s2a )
          v53 = (unsigned __int64)s2a;
        v48 = &nb[v53];
        if ( v97 < v53 )
        {
          v49 = 0;
          goto LABEL_49;
        }
LABEL_135:
        if ( v52 <= v49 )
          v49 = v52;
        v49 -= v53;
        goto LABEL_49;
      }
    }
    else
    {
      v91 = memchr(nb, 91, (size_t)s2a);
      v48 = nb;
      v49 = (size_t)s2a;
      v52 = -1;
      if ( v91 )
      {
        v52 = -1;
        v53 = v91 - nb + 1;
        if ( v53 > (unsigned __int64)s2a )
          v53 = (unsigned __int64)s2a;
        v48 = &nb[v53];
        goto LABEL_135;
      }
    }
    if ( v49 > v52 )
      v49 = v52;
    goto LABEL_49;
  }
  return result;
}
