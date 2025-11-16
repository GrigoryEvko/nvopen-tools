// Function: sub_978F70
// Address: 0x978f70
//
__int64 __fastcall sub_978F70(
        char *s1,
        unsigned __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  unsigned int v9; // r14d
  __int64 v12; // r15
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 *v16; // r9
  _QWORD *v17; // rcx
  _BYTE *v18; // r11
  size_t v19; // r10
  _QWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  char v25; // al
  __int64 v26; // r15
  _BYTE *v27; // rcx
  unsigned int v28; // ebx
  __int64 *v29; // r9
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rsi
  _BYTE *v37; // rdx
  _BYTE *v38; // rdx
  __int64 v39; // rax
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // r8
  __int64 v43; // rax
  __int64 v44; // rdi
  _BYTE *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  _BYTE *v49; // rdx
  __int64 v50; // rcx
  _BYTE *v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r8
  bool v54; // al
  bool v55; // al
  __int64 v56; // rax
  int v57; // eax
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // [rsp+8h] [rbp-A8h]
  size_t nb; // [rsp+10h] [rbp-A0h]
  size_t nc; // [rsp+10h] [rbp-A0h]
  size_t n; // [rsp+10h] [rbp-A0h]
  size_t na; // [rsp+10h] [rbp-A0h]
  void *srce; // [rsp+18h] [rbp-98h]
  _BYTE *srcf; // [rsp+18h] [rbp-98h]
  __int64 *srcg; // [rsp+18h] [rbp-98h]
  void *src; // [rsp+18h] [rbp-98h]
  _BYTE *srca; // [rsp+18h] [rbp-98h]
  void *srcb; // [rsp+18h] [rbp-98h]
  __int64 *srcc; // [rsp+18h] [rbp-98h]
  void *srch; // [rsp+18h] [rbp-98h]
  __int64 *srcd; // [rsp+18h] [rbp-98h]
  void *srci; // [rsp+18h] [rbp-98h]
  __int64 v75; // [rsp+20h] [rbp-90h]
  __int64 v76; // [rsp+20h] [rbp-90h]
  __int64 v77; // [rsp+20h] [rbp-90h]
  __int64 *v78; // [rsp+20h] [rbp-90h]
  _QWORD *v79; // [rsp+20h] [rbp-90h]
  _QWORD *v80; // [rsp+20h] [rbp-90h]
  _BYTE *v81; // [rsp+20h] [rbp-90h]
  __int64 v82; // [rsp+20h] [rbp-90h]
  _BYTE *v83; // [rsp+20h] [rbp-90h]
  __int64 v84; // [rsp+20h] [rbp-90h]
  __int64 *v85; // [rsp+28h] [rbp-88h]
  __int64 v86; // [rsp+28h] [rbp-88h]
  __int64 *v87; // [rsp+28h] [rbp-88h]
  void *v88; // [rsp+28h] [rbp-88h]
  __int64 v89; // [rsp+28h] [rbp-88h]
  _BYTE *v90; // [rsp+28h] [rbp-88h]
  __int64 v91; // [rsp+28h] [rbp-88h]
  __int64 v92; // [rsp+28h] [rbp-88h]
  __int64 v93; // [rsp+28h] [rbp-88h]
  _BYTE *v94; // [rsp+28h] [rbp-88h]
  __int64 v95; // [rsp+28h] [rbp-88h]
  size_t v96; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v97[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v98[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v99; // [rsp+60h] [rbp-50h]
  __int64 v100; // [rsp+68h] [rbp-48h]
  __int64 v101; // [rsp+70h] [rbp-40h]

  v9 = a3;
  if ( a6 == 1 )
    return sub_975D30(s1, a2, a3, a4, a5, a7, a8);
  if ( a6 == 2 )
  {
    v14 = sub_B43CA0(a8);
    v15 = a8;
    v16 = a7;
    v17 = (_QWORD *)v14;
    v97[0] = v98;
    v18 = *(_BYTE **)(v14 + 232);
    v19 = *(_QWORD *)(v14 + 240);
    if ( &v18[v19] && !v18 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v96 = *(_QWORD *)(v14 + 240);
    if ( v19 > 0xF )
    {
      nb = v19;
      srcf = v18;
      v79 = (_QWORD *)v14;
      v33 = sub_22409D0(v97, &v96, 0);
      v17 = v79;
      v18 = srcf;
      v97[0] = v33;
      v34 = (_QWORD *)v33;
      v19 = nb;
      v16 = a7;
      v98[0] = v96;
      v15 = a8;
    }
    else
    {
      if ( v19 == 1 )
      {
        LOBYTE(v98[0]) = *v18;
        v20 = v98;
        goto LABEL_12;
      }
      if ( !v19 )
      {
        v20 = v98;
        goto LABEL_12;
      }
      v34 = v98;
    }
    nc = v15;
    srcg = v16;
    v80 = v17;
    memcpy(v34, v18, v19);
    v19 = v96;
    v20 = (_QWORD *)v97[0];
    v17 = v80;
    v16 = srcg;
    v15 = nc;
LABEL_12:
    v97[1] = v19;
    *((_BYTE *)v20 + v19) = 0;
    v99 = v17[33];
    v100 = v17[34];
    v101 = v17[35];
    if ( (unsigned int)(v99 - 42) > 1 )
    {
      if ( (_QWORD *)v97[0] != v98 )
      {
        v77 = v15;
        v87 = v16;
        j_j___libc_free_0(v97[0], v98[0] + 1LL);
        v16 = v87;
        v15 = v77;
      }
    }
    else
    {
      if ( (_QWORD *)v97[0] != v98 )
      {
        v75 = v15;
        v85 = v16;
        j_j___libc_free_0(v97[0], v98[0] + 1LL);
        v16 = v85;
        v15 = v75;
      }
      v21 = *a5;
      if ( *(_BYTE *)*a5 == 18 )
      {
        v22 = a5[1];
        if ( *(_BYTE *)v22 == 18 && *(_QWORD *)(v22 + 8) == *(_QWORD *)(v21 + 8) && s1[1] == 90 && a2 > 6 )
        {
          v86 = v22 + 24;
          v76 = v21 + 24;
          if ( a2 == 8 )
          {
            if ( *(_QWORD *)s1 == 0x6666776F70335A5FLL || *(_QWORD *)s1 == 0x6464776F70335A5FLL )
            {
              srce = (void *)v15;
              v23 = sub_96A630((double (__fastcall *)(double, double))&pow, v76, v86, a4);
              v15 = (__int64)srce;
              v12 = v23;
              goto LABEL_24;
            }
          }
          else if ( a2 == 9 )
          {
            na = v15;
            srcd = v16;
            v57 = memcmp(s1, "_Z4fmodff", 9u);
            v15 = na;
            if ( !v57 || (v58 = memcmp(s1, "_Z4fmoddd", 9u), v16 = srcd, v15 = na, !v58) )
            {
              srci = (void *)v15;
              v59 = sub_96A630((double (__fastcall *)(double, double))&fmod, v76, v86, a4);
              v15 = (__int64)srci;
              v12 = v59;
LABEL_24:
              if ( v12 )
                return v12;
              return sub_9732F0(v9, a4, a5, v15);
            }
          }
          n = v15;
          srcc = v16;
          v54 = sub_9691B0(s1, a2, "_Z5atan2ff", 10);
          v15 = n;
          if ( v54 || (v55 = sub_9691B0(s1, a2, "_Z5atan2dd", 10), v16 = srcc, v15 = n, v55) )
          {
            srch = (void *)v15;
            v56 = sub_96A630((double (__fastcall *)(double, double))&atan2, v76, v86, a4);
            v15 = (__int64)srch;
            v12 = v56;
            goto LABEL_24;
          }
        }
      }
    }
    if ( !v16 )
      return sub_9732F0(v9, a4, a5, v15);
    v24 = *v16;
    v88 = (void *)v15;
    LODWORD(v96) = 524;
    v78 = v16;
    v25 = sub_980AF0(v24, s1, a2, &v96);
    v15 = (__int64)v88;
    if ( !v25 )
      return sub_9732F0(v9, a4, a5, v15);
    v26 = *a5;
    if ( *(_BYTE *)*a5 != 18 )
      return sub_9732F0(v9, a4, a5, v15);
    v27 = (_BYTE *)a5[1];
    if ( *v27 != 18 )
      return sub_9732F0(v9, a4, a5, v15);
    v28 = v96;
    v29 = v78;
    v30 = v26 + 24;
    v31 = (__int64)(v27 + 24);
    if ( (unsigned int)v96 > 0x115 )
    {
      if ( (unsigned int)v96 <= 0x183 )
      {
        if ( (unsigned int)v96 <= 0x181 )
          return sub_9732F0(v9, a4, a5, v15);
LABEL_64:
        if ( (v78[((unsigned __int64)(unsigned int)v96 >> 6) + 1] & (1LL << v96)) == 0
          && (((int)*(unsigned __int8 *)(*v78 + ((unsigned int)v96 >> 2)) >> (2 * (v96 & 3))) & 3) != 0 )
        {
          v46 = sub_96A630((double (__fastcall *)(double, double))&pow, v30, v31, a4);
          v15 = (__int64)v88;
          v12 = v46;
          goto LABEL_24;
        }
        return sub_9732F0(v9, a4, a5, v15);
      }
      if ( (unsigned int)(v96 - 404) > 1 )
        return sub_9732F0(v9, a4, a5, v15);
      v81 = v27 + 24;
      if ( (v29[7] & (1LL << v96)) != 0 || (((int)*(unsigned __int8 *)(*v29 + 101) >> (2 * (v96 & 3))) & 3) == 0 )
        return sub_9732F0(v9, a4, a5, v15);
      src = v88;
      v35 = sub_C33340(v24, v30, v31, 2 * (unsigned int)(v96 & 3), v88);
      v36 = v26 + 24;
      v37 = v81;
      v82 = v35;
      v90 = v37;
      if ( *(_QWORD *)(v26 + 24) == v35 )
      {
        sub_C3C790(v97, v36);
        v40 = (__int64)src;
        v39 = v82;
        v38 = v90;
      }
      else
      {
        sub_C33EB0(v97, v36);
        v38 = v90;
        v39 = v82;
        v40 = (__int64)src;
      }
      v91 = v40;
      if ( v97[0] == v39 )
        v41 = sub_C3E9B0(v97, v38);
      else
        v41 = sub_C3C0A0(v97, v38);
      v42 = v91;
    }
    else
    {
      if ( (unsigned int)v96 <= 0x113 )
      {
        if ( (unsigned int)v96 > 0x7F )
        {
          if ( (unsigned int)(v96 - 174) > 1 )
            return sub_9732F0(v9, a4, a5, v15);
          v60 = (__int64)v88;
          srca = v27 + 24;
          v93 = a5[1];
          v43 = sub_C33340(v24, v30, v31, v27, v15);
          v30 = v26 + 24;
          v31 = (__int64)srca;
          v29 = v78;
          v15 = v60;
          v44 = v26 + 24;
          if ( *(_QWORD *)(v26 + 24) == v43 )
            v44 = *(_QWORD *)(v26 + 32);
          if ( (*(_BYTE *)(v44 + 20) & 7) == 3 )
          {
            v45 = srca;
            if ( v43 == *(_QWORD *)(v93 + 24) )
              v45 = *(_BYTE **)(v93 + 32);
            if ( (v45[20] & 7) == 3 )
              return sub_9732F0(v9, a4, a5, v15);
          }
LABEL_37:
          if ( (v29[((unsigned __int64)v28 >> 6) + 1] & (1LL << v28)) == 0
            && (((int)*(unsigned __int8 *)(*v29 + (v28 >> 2)) >> (2 * (v28 & 3))) & 3) != 0 )
          {
            v89 = v15;
            v32 = sub_96A630((double (__fastcall *)(double, double))&atan2, v30, v31, a4);
            v15 = v89;
            v12 = v32;
            goto LABEL_24;
          }
          return sub_9732F0(v9, a4, a5, v15);
        }
        if ( (unsigned int)v96 <= 0x7D )
        {
          if ( (unsigned int)(v96 - 75) > 1 )
            return sub_9732F0(v9, a4, a5, v15);
          goto LABEL_37;
        }
        goto LABEL_64;
      }
      v83 = v27 + 24;
      if ( (v29[5] & (1LL << v96)) != 0 || (((int)*(unsigned __int8 *)(*v29 + 69) >> (2 * (v96 & 3))) & 3) == 0 )
        return sub_9732F0(v9, a4, a5, v15);
      srcb = v88;
      v47 = sub_C33340(v24, v30, v31, 2 * (unsigned int)(v96 & 3), v88);
      v48 = v26 + 24;
      v49 = v83;
      v84 = v47;
      v94 = v49;
      if ( *(_QWORD *)(v26 + 24) == v47 )
      {
        sub_C3C790(v97, v48);
        v53 = (__int64)srcb;
        v52 = v84;
        v51 = v94;
      }
      else
      {
        sub_C33EB0(v97, v48);
        v51 = v94;
        v52 = v84;
        v53 = (__int64)srcb;
      }
      v95 = v53;
      if ( v97[0] == v52 )
        v41 = sub_C3EC80(v97, v51, v51, v50);
      else
        v41 = sub_C3BE30(v97, v51, v51, v50);
      v42 = v95;
    }
    v92 = v42;
    if ( !v41 )
    {
      v12 = sub_AC8EA0(*a4, v97);
      sub_91D830(v97);
      v15 = v92;
      goto LABEL_24;
    }
    sub_91D830(v97);
    v15 = v92;
    return sub_9732F0(v9, a4, a5, v15);
  }
  v12 = 0;
  if ( a6 != 3 )
    return v12;
  return sub_96AF60((unsigned int)a3, a4, (unsigned __int8 **)a5, a8);
}
