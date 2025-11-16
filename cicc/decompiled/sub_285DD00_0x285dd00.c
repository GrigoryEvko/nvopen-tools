// Function: sub_285DD00
// Address: 0x285dd00
//
_QWORD *__fastcall sub_285DD00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // r8
  unsigned __int8 v8; // bl
  __int64 v9; // r9
  __int64 v10; // r15
  unsigned int v11; // r14d
  __int64 v12; // r10
  int v13; // eax
  bool v14; // al
  int v15; // eax
  __int16 v16; // ax
  __int64 v17; // r12
  int v18; // ebx
  __int64 v19; // r10
  _QWORD *result; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rsi
  void *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // r15
  _QWORD *v29; // r12
  __int64 v30; // rdx
  unsigned __int64 v31; // r9
  __int64 v32; // rdi
  __int64 v33; // r15
  __int64 *v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rcx
  _BYTE *v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r9
  _BYTE *v40; // rax
  __int64 v41; // rcx
  _BYTE *v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r11
  size_t v46; // rdx
  int v47; // eax
  _QWORD *v48; // rcx
  __int64 *v49; // r15
  __int64 *v50; // rbx
  bool v51; // r10
  __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  int v56; // r14d
  _QWORD *v57; // rax
  __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // rax
  int v61; // r14d
  _QWORD *v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  bool v65; // al
  bool v66; // al
  unsigned __int8 v67; // [rsp+Fh] [rbp-D1h]
  unsigned __int8 v68; // [rsp+10h] [rbp-D0h]
  unsigned __int8 v69; // [rsp+10h] [rbp-D0h]
  __int64 v70; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v71; // [rsp+18h] [rbp-C8h]
  __int64 v72; // [rsp+18h] [rbp-C8h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  __int64 v74; // [rsp+18h] [rbp-C8h]
  unsigned int v75; // [rsp+20h] [rbp-C0h]
  unsigned int v76; // [rsp+20h] [rbp-C0h]
  __int64 v77; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v78; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v79; // [rsp+20h] [rbp-C0h]
  unsigned int v80; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v81; // [rsp+20h] [rbp-C0h]
  __int64 v82; // [rsp+28h] [rbp-B8h]
  __int64 v83; // [rsp+28h] [rbp-B8h]
  __int64 v84; // [rsp+28h] [rbp-B8h]
  __int64 v85; // [rsp+28h] [rbp-B8h]
  _QWORD *v86; // [rsp+28h] [rbp-B8h]
  _QWORD *v87; // [rsp+28h] [rbp-B8h]
  __int64 v88; // [rsp+28h] [rbp-B8h]
  __int64 v89; // [rsp+28h] [rbp-B8h]
  _QWORD *v90; // [rsp+28h] [rbp-B8h]
  __int64 v91; // [rsp+28h] [rbp-B8h]
  __int64 v92; // [rsp+28h] [rbp-B8h]
  __int64 v93; // [rsp+28h] [rbp-B8h]
  __int64 v94; // [rsp+28h] [rbp-B8h]
  _QWORD *v95; // [rsp+28h] [rbp-B8h]
  __int64 v96; // [rsp+28h] [rbp-B8h]
  __int64 v97; // [rsp+28h] [rbp-B8h]
  bool v98; // [rsp+28h] [rbp-B8h]
  _QWORD *v99; // [rsp+28h] [rbp-B8h]
  void *s1; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+38h] [rbp-A8h]
  _BYTE v102[32]; // [rsp+40h] [rbp-A0h] BYREF
  void *s2; // [rsp+60h] [rbp-80h] BYREF
  __int64 v104; // [rsp+68h] [rbp-78h]
  _QWORD v105[14]; // [rsp+70h] [rbp-70h] BYREF

  if ( a1 == a2 )
  {
    v25 = sub_D95540(a1);
    return sub_DA2C50((__int64)a3, v25, 1, 0);
  }
  v6 = *(unsigned __int16 *)(a2 + 24);
  v7 = a2;
  v8 = a4;
  v9 = (unsigned int)a4;
  if ( (_WORD)v6 )
  {
    v16 = *(_WORD *)(a1 + 24);
    if ( !v16 )
      return 0;
LABEL_16:
    if ( v16 == 8 )
    {
      if ( v8
        || (v93 = v7,
            v55 = sub_D95540(**(_QWORD **)(a1 + 32)),
            v56 = sub_D97050((__int64)a3, v55),
            v57 = (_QWORD *)sub_B2BE50(*a3),
            v58 = sub_BCCE00(v57, v56 + 1),
            v59 = sub_DC5000((__int64)a3, a1, v58, 0),
            v7 = v93,
            *((_WORD *)v59 + 12) == 8) )
      {
        if ( *(_QWORD *)(a1 + 40) == 2 )
        {
          v84 = v7;
          v21 = sub_D33D80((_QWORD *)a1, (__int64)a3, v6, a4, v7);
          v22 = sub_285DD00(v21, v84, a3, v8, v84);
          if ( v22 )
          {
            v23 = sub_285DD00(**(_QWORD **)(a1 + 32), v84, a3, v8, v84);
            if ( v23 )
              return sub_DC1960((__int64)a3, v23, v22, *(_QWORD *)(a1 + 48), 0);
          }
        }
      }
      return 0;
    }
    if ( v16 == 5 )
    {
      if ( !v8 )
      {
        v96 = v7;
        v61 = sub_D97050((__int64)a3, *(_QWORD *)(a1 + 48));
        v62 = (_QWORD *)sub_B2BE50(*a3);
        v63 = sub_BCCE00(v62, v61 + 1);
        v64 = sub_DC5000((__int64)a3, a1, v63, 0);
        v7 = v96;
        if ( *((_WORD *)v64 + 12) != 5 )
          return 0;
      }
      v26 = *(_QWORD *)(a1 + 40);
      v104 = 0x800000000LL;
      v27 = *(_QWORD **)(a1 + 32);
      s2 = v105;
      v28 = &v27[v26];
      if ( v28 == v27 )
      {
LABEL_45:
        result = sub_DC7EB0(a3, (__int64)&s2, 0, 0);
      }
      else
      {
        v29 = v27;
        while ( 1 )
        {
          v88 = v7;
          result = (_QWORD *)sub_285DD00(*v29, v7, a3, v8, v7);
          if ( !result )
            break;
          v30 = (unsigned int)v104;
          v7 = v88;
          v31 = (unsigned int)v104 + 1LL;
          if ( v31 > HIDWORD(v104) )
          {
            v73 = v88;
            v95 = result;
            sub_C8D5F0((__int64)&s2, v105, (unsigned int)v104 + 1LL, 8u, v7, v31);
            v30 = (unsigned int)v104;
            v7 = v73;
            result = v95;
          }
          ++v29;
          *((_QWORD *)s2 + v30) = result;
          LODWORD(v104) = v104 + 1;
          if ( v28 == v29 )
            goto LABEL_45;
        }
      }
      goto LABEL_46;
    }
    if ( v16 != 6 )
      return 0;
    if ( v8 )
    {
      if ( (_WORD)v6 == 6 )
        goto LABEL_51;
    }
    else
    {
      v81 = v9;
      v97 = v7;
      v65 = sub_284F530(a1, a3);
      v7 = v97;
      v9 = v81;
      if ( !v65 )
        return 0;
      if ( *(_WORD *)(v97 + 24) == 6 )
      {
        v66 = sub_284F530(v97, a3);
        v7 = v97;
        v9 = v81;
        if ( v66 )
        {
LABEL_51:
          v32 = *(_QWORD *)(a1 + 32);
          v33 = *(_QWORD *)v32;
          if ( !*(_WORD *)(*(_QWORD *)v32 + 24LL) )
          {
            v34 = *(__int64 **)(v7 + 32);
            if ( !*(_WORD *)(*v34 + 24) )
            {
              v71 = v9;
              v77 = *v34;
              v89 = v7;
              v35 = (_BYTE *)sub_2850B10(v32, *(_QWORD *)(a1 + 40), 1);
              v101 = v36;
              s1 = v102;
              sub_D9CA60((__int64)&s1, v35, v37, v36, v38, v39);
              v40 = (_BYTE *)sub_2850B10(*(_QWORD *)(v89 + 32), *(_QWORD *)(v89 + 40), 1);
              s2 = v105;
              v104 = v41;
              sub_D9CA60((__int64)&s2, v40, v42, v41, v43, v44);
              v7 = v89;
              v45 = v77;
              v9 = v71;
              if ( (unsigned int)v101 != (unsigned __int64)(unsigned int)v104 )
              {
                v48 = s2;
LABEL_56:
                if ( v48 != v105 )
                {
                  v78 = v9;
                  v91 = v7;
                  _libc_free((unsigned __int64)v48);
                  v9 = v78;
                  v7 = v91;
                }
                if ( s1 != v102 )
                {
                  v79 = v9;
                  v92 = v7;
                  _libc_free((unsigned __int64)s1);
                  v9 = v79;
                  v7 = v92;
                }
                v32 = *(_QWORD *)(a1 + 32);
                goto LABEL_61;
              }
              v67 = v71;
              v72 = v89;
              v46 = 8LL * (unsigned int)v101;
              if ( v46 )
              {
                v90 = s2;
                v47 = memcmp(s1, s2, v46);
                v48 = v90;
                v45 = v77;
                v7 = v72;
                v9 = v67;
                if ( v47 )
                  goto LABEL_56;
              }
              result = (_QWORD *)sub_285DD00(v33, v45, a3, v8, v7);
              if ( s2 != v105 )
              {
                v99 = result;
                _libc_free((unsigned __int64)s2);
                result = v99;
              }
              v24 = s1;
              if ( s1 == v102 )
                return result;
LABEL_24:
              v86 = result;
              _libc_free((unsigned __int64)v24);
              return v86;
            }
          }
LABEL_61:
          s2 = v105;
          v104 = 0x400000000LL;
          v49 = (__int64 *)(v32 + 8LL * *(_QWORD *)(a1 + 40));
          result = 0;
          if ( (__int64 *)v32 == v49 )
            return result;
          v50 = (__int64 *)v32;
          v51 = 0;
          v80 = (unsigned __int8)v9;
          do
          {
            v52 = *v50;
            if ( !v51 )
            {
              v94 = v7;
              v60 = sub_285DD00(*v50, v7, a3, v80, v7);
              v7 = v94;
              if ( v60 )
                v52 = v60;
              v51 = v60 != 0;
            }
            v53 = (unsigned int)v104;
            v54 = (unsigned int)v104 + 1LL;
            if ( v54 > HIDWORD(v104) )
            {
              v74 = v7;
              v98 = v51;
              sub_C8D5F0((__int64)&s2, v105, v54, 8u, v7, v9);
              v53 = (unsigned int)v104;
              v7 = v74;
              v51 = v98;
            }
            ++v50;
            *((_QWORD *)s2 + v53) = v52;
            LODWORD(v104) = v104 + 1;
          }
          while ( v49 != v50 );
          result = 0;
          if ( v51 )
            result = sub_DC8BD0(a3, (__int64)&s2, 0, 0);
LABEL_46:
          v24 = s2;
          if ( s2 == v105 )
            return result;
          goto LABEL_24;
        }
      }
    }
    v32 = *(_QWORD *)(a1 + 32);
    goto LABEL_61;
  }
  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(_DWORD *)(v10 + 32);
  if ( !v11 )
    goto LABEL_22;
  v12 = v10 + 24;
  if ( v11 <= 0x40 )
  {
    a4 = 64 - v11;
    v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)(v10 + 24);
  }
  else
  {
    v75 = *(unsigned __int16 *)(a2 + 24);
    v68 = a4;
    v13 = sub_C445E0(v10 + 24);
    v12 = v10 + 24;
    v6 = v75;
    v7 = a2;
    v9 = v68;
    v14 = v11 == v13;
  }
  if ( v14 )
  {
LABEL_22:
    v85 = v7;
    if ( *(_BYTE *)(sub_D95540(a1) + 8) == 14 )
      return 0;
    v105[0] = a1;
    s2 = v105;
    v105[1] = v85;
    v104 = 0x200000002LL;
    result = sub_DC8BD0(a3, (__int64)&s2, 0, 0);
    v24 = s2;
    if ( s2 == v105 )
      return result;
    goto LABEL_24;
  }
  if ( v11 <= 0x40 )
  {
    if ( *(_QWORD *)(v10 + 24) != 1 )
      goto LABEL_10;
    return (_QWORD *)a1;
  }
  v69 = v9;
  v70 = v7;
  v76 = v6;
  v82 = v12;
  v15 = sub_C444A0(v12);
  v12 = v82;
  v6 = v76;
  v7 = v70;
  v9 = v69;
  if ( v11 - v15 <= 0x40 && **(_QWORD **)(v10 + 24) == 1 )
    return (_QWORD *)a1;
LABEL_10:
  v16 = *(_WORD *)(a1 + 24);
  if ( v16 )
    goto LABEL_16;
  v83 = v12;
  v17 = *(_QWORD *)(a1 + 32) + 24LL;
  sub_C4B8A0((__int64)&s2, v17, v12);
  v18 = v104;
  v19 = v83;
  if ( (unsigned int)v104 > 0x40 )
  {
    if ( v18 - (unsigned int)sub_C444A0((__int64)&s2) > 0x40 || *(_QWORD *)s2 )
    {
      if ( s2 )
        j_j___libc_free_0_0((unsigned __int64)s2);
      return 0;
    }
    j_j___libc_free_0_0((unsigned __int64)s2);
    v19 = v83;
  }
  else if ( s2 )
  {
    return 0;
  }
  sub_C4A3E0((__int64)&s2, v17, v19);
  result = sub_DA26C0(a3, (__int64)&s2);
  if ( (unsigned int)v104 > 0x40 )
  {
    if ( s2 )
    {
      v87 = result;
      j_j___libc_free_0_0((unsigned __int64)s2);
      return v87;
    }
  }
  return result;
}
