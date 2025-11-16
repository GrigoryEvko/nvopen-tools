// Function: sub_ACB110
// Address: 0xacb110
//
__int64 __fastcall sub_ACB110(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  int v6; // eax
  _QWORD *v7; // rax
  void **p_src; // rdi
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 result; // rax
  void *v13; // rdx
  _QWORD *v14; // r14
  _BYTE *v15; // rax
  _QWORD *v16; // rbx
  size_t v17; // rax
  __int64 v18; // rax
  size_t v19; // r12
  char *v20; // r13
  __int64 v21; // rax
  __int64 **v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  void *v25; // rdi
  unsigned __int8 v26; // al
  _QWORD *v27; // rbx
  __int64 v28; // rax
  size_t v29; // rax
  size_t v30; // rdx
  void *v31; // rdx
  void *v32; // rdx
  _QWORD *v33; // r14
  _BYTE *v34; // rax
  _QWORD *v35; // r15
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r12
  char *v40; // r14
  size_t v41; // r13
  __int64 v42; // rdi
  __int64 **v43; // rax
  _QWORD *v44; // r14
  _BYTE *v45; // rax
  _QWORD *v46; // rbx
  size_t v47; // rax
  __int64 v48; // rax
  size_t v49; // r12
  char *v50; // r14
  __int64 v51; // rax
  __int64 **v52; // rax
  _QWORD *v53; // r14
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  int v57; // eax
  void *v58; // rdx
  _QWORD *v59; // r15
  _BYTE *v60; // rax
  _QWORD *v61; // r14
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // r14
  __int64 v66; // rax
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  int v69; // eax
  int v70; // eax
  __int64 v71; // [rsp+8h] [rbp-D8h]
  _BYTE *v72; // [rsp+8h] [rbp-D8h]
  unsigned int v73; // [rsp+8h] [rbp-D8h]
  _BYTE *v74; // [rsp+8h] [rbp-D8h]
  unsigned int v75; // [rsp+8h] [rbp-D8h]
  _BYTE *v76; // [rsp+8h] [rbp-D8h]
  unsigned int v77; // [rsp+8h] [rbp-D8h]
  unsigned int v78; // [rsp+8h] [rbp-D8h]
  unsigned int v79; // [rsp+8h] [rbp-D8h]
  __int64 v80; // [rsp+8h] [rbp-D8h]
  void **v81; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v82; // [rsp+18h] [rbp-C8h]
  void *src; // [rsp+20h] [rbp-C0h] BYREF
  size_t n; // [rsp+28h] [rbp-B8h]
  size_t v85; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE v86[168]; // [rsp+38h] [rbp-A8h] BYREF

  if ( !a3 )
    return sub_AC9350(a1);
  v3 = *(_QWORD *)a2;
  v4 = (_QWORD *)a2;
  v5 = (_QWORD *)(a2 + 8 * a3);
  v6 = **(unsigned __int8 **)a2;
  if ( (_BYTE)v6 == 13 )
  {
    if ( v5 != (_QWORD *)a2 )
    {
      v23 = (_QWORD *)a2;
      while ( ++v23 != v5 )
      {
        if ( v3 != *v23 )
          goto LABEL_24;
      }
    }
    return sub_ACADE0(a1);
  }
  if ( (unsigned int)(v6 - 12) <= 1 )
  {
    if ( v5 != (_QWORD *)a2 )
    {
LABEL_24:
      v24 = (_QWORD *)a2;
      while ( v5 != ++v24 )
      {
        if ( v3 != *v24 )
        {
          if ( sub_AC30F0(*(_QWORD *)a2) )
            goto LABEL_6;
          goto LABEL_9;
        }
      }
    }
    return sub_ACA8A0(a1);
  }
  if ( sub_AC30F0(*(_QWORD *)a2) )
  {
    if ( v5 != (_QWORD *)a2 )
    {
LABEL_6:
      v7 = (_QWORD *)a2;
      while ( v3 == *v7 )
      {
        if ( v5 == ++v7 )
          return sub_AC9350(a1);
      }
      goto LABEL_9;
    }
    return sub_AC9350(a1);
  }
LABEL_9:
  p_src = *(void ***)(v3 + 8);
  LOBYTE(v9) = sub_AC5240((__int64)p_src);
  v11 = v9;
  result = 0;
  if ( !(_BYTE)v11 )
    return result;
  if ( *(_BYTE *)v3 == 17 )
  {
    a2 = 8;
    if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(v3 + 8), 8) )
    {
      v14 = v4;
      n = 0;
      src = v86;
      v85 = 16;
      if ( v5 == v4 )
      {
LABEL_19:
        v18 = sub_BD5C60(*v4, a2, v13);
        v19 = n;
        v20 = (char *)src;
        v21 = sub_BCD140(v18, 8);
        v22 = (__int64 **)sub_BCD420(v21, v19);
        a2 = v19;
        result = sub_AC9630(v20, v19, v22);
        goto LABEL_31;
      }
      while ( 1 )
      {
        v15 = (_BYTE *)*v14;
        if ( *(_BYTE *)*v14 != 17 )
          goto LABEL_30;
        v16 = (_QWORD *)*((_QWORD *)v15 + 3);
        if ( *((_DWORD *)v15 + 8) > 0x40u )
          v16 = (_QWORD *)*v16;
        v17 = n;
        if ( n + 1 > v85 )
        {
          a2 = (__int64)v86;
          sub_C8D290(&src, v86, n + 1, 1);
          v17 = n;
        }
        v13 = src;
        ++v14;
        *((_BYTE *)src + v17) = (_BYTE)v16;
        ++n;
        if ( v5 == v14 )
          goto LABEL_19;
      }
    }
    a2 = 16;
    if ( !(unsigned __int8)sub_BCAC40(*(_QWORD *)(v3 + 8), 16) )
    {
      a2 = 32;
      if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(v3 + 8), 32) )
      {
        v33 = v4;
        src = &v85;
        n = 0x1000000000LL;
        if ( v5 != v4 )
        {
          while ( 1 )
          {
            v34 = (_BYTE *)*v33;
            if ( *(_BYTE *)*v33 != 17 )
              break;
            v35 = (_QWORD *)*((_QWORD *)v34 + 3);
            if ( *((_DWORD *)v34 + 8) > 0x40u )
              v35 = (_QWORD *)*v35;
            v36 = (unsigned int)n;
            v37 = (unsigned int)n + 1LL;
            if ( v37 > HIDWORD(n) )
            {
              a2 = (__int64)&v85;
              sub_C8D5F0(&src, &v85, v37, 4);
              v36 = (unsigned int)n;
            }
            v32 = src;
            ++v33;
            *((_DWORD *)src + v36) = (_DWORD)v35;
            LODWORD(n) = n + 1;
            if ( v5 == v33 )
              goto LABEL_64;
          }
LABEL_91:
          result = 0;
          goto LABEL_66;
        }
LABEL_64:
        v38 = sub_BD5C60(*v4, a2, v32);
        v39 = (unsigned int)n;
        v40 = (char *)src;
        v41 = 4LL * (unsigned int)n;
        v42 = sub_BCD140(v38, 32);
        goto LABEL_65;
      }
      a2 = 64;
      if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(v3 + 8), 64) )
      {
        v59 = v4;
        src = &v85;
        for ( n = 0x1000000000LL; v5 != v59; LODWORD(n) = n + 1 )
        {
          v60 = (_BYTE *)*v59;
          if ( *(_BYTE *)*v59 != 17 )
            goto LABEL_91;
          v61 = (_QWORD *)*((_QWORD *)v60 + 3);
          if ( *((_DWORD *)v60 + 8) > 0x40u )
            v61 = (_QWORD *)*v61;
          v62 = (unsigned int)n;
          v63 = (unsigned int)n + 1LL;
          if ( v63 > HIDWORD(n) )
          {
            a2 = (__int64)&v85;
            sub_C8D5F0(&src, &v85, v63, 8);
            v62 = (unsigned int)n;
          }
          v58 = src;
          ++v59;
          *((_QWORD *)src + v62) = v61;
        }
        v64 = sub_BD5C60(*v4, a2, v58);
        v39 = (unsigned int)n;
        v40 = (char *)src;
        v41 = 8LL * (unsigned int)n;
        v42 = sub_BCD140(v64, 64);
LABEL_65:
        v43 = (__int64 **)sub_BCD420(v42, v39);
        a2 = v41;
        result = sub_AC9630(v40, v41, v43);
        goto LABEL_66;
      }
      return 0;
    }
    v44 = v4;
    n = 0;
    src = v86;
    v85 = 16;
    if ( v5 != v4 )
    {
      while ( 1 )
      {
        v45 = (_BYTE *)*v44;
        if ( *(_BYTE *)*v44 != 17 )
          break;
        v46 = (_QWORD *)*((_QWORD *)v45 + 3);
        if ( *((_DWORD *)v45 + 8) > 0x40u )
          v46 = (_QWORD *)*v46;
        v47 = n;
        if ( n + 1 > v85 )
        {
          a2 = (__int64)v86;
          sub_C8D290(&src, v86, n + 1, 2);
          v47 = n;
        }
        v31 = src;
        ++v44;
        *((_WORD *)src + v47) = (_WORD)v46;
        ++n;
        if ( v5 == v44 )
          goto LABEL_75;
      }
LABEL_30:
      result = 0;
      goto LABEL_31;
    }
LABEL_75:
    v48 = sub_BD5C60(*v4, a2, v31);
    v49 = n;
    v50 = (char *)src;
    v51 = sub_BCD140(v48, 16);
    v52 = (__int64 **)sub_BCD420(v51, v49);
    a2 = 2 * v49;
    result = sub_AC9630(v50, 2 * v49, v52);
    goto LABEL_31;
  }
  if ( *(_BYTE *)v3 != 18 )
    return 0;
  v26 = *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL);
  if ( v26 <= 1u )
  {
    n = 0;
    src = v86;
    v85 = 16;
    if ( v5 != (_QWORD *)a2 )
    {
      v27 = (_QWORD *)a2;
      while ( 1 )
      {
        v72 = (_BYTE *)*v27;
        if ( *(_BYTE *)*v27 != 18 )
          goto LABEL_30;
        v28 = sub_C33340(p_src, a2, *v27, v10, v11);
        p_src = (void **)&v81;
        a2 = (__int64)(v72 + 24);
        if ( *((_QWORD *)v72 + 3) == v28 )
          sub_C3E660(&v81, a2);
        else
          sub_C3A850(&v81, a2);
        v73 = v82;
        if ( v82 > 0x40 )
        {
          p_src = (void **)&v81;
          v57 = sub_C444A0(&v81);
          v11 = 0xFFFFFFFFLL;
          if ( v73 - v57 <= 0x40 )
            v11 = *(unsigned __int16 *)v81;
        }
        else
        {
          v11 = (unsigned __int16)v81;
        }
        v29 = n;
        if ( n + 1 > v85 )
        {
          p_src = &src;
          a2 = (__int64)v86;
          v78 = v11;
          sub_C8D290(&src, v86, n + 1, 2);
          v29 = n;
          v11 = v78;
        }
        *((_WORD *)src + v29) = v11;
        ++n;
        if ( v82 > 0x40 )
        {
          p_src = v81;
          if ( v81 )
            j_j___libc_free_0_0(v81);
        }
        if ( v5 == ++v27 )
        {
          a2 = (__int64)src;
          v30 = n;
          goto LABEL_52;
        }
      }
    }
    v30 = 0;
    a2 = (__int64)v86;
LABEL_52:
    result = sub_AC9910(*(_QWORD *)(*v4 + 8LL), (char *)a2, v30);
LABEL_31:
    v25 = src;
    if ( src == v86 )
      return result;
    goto LABEL_32;
  }
  if ( v26 != 2 )
  {
    if ( v26 == 3 )
    {
      v53 = (_QWORD *)a2;
      src = &v85;
      n = 0x1000000000LL;
      if ( v5 == (_QWORD *)a2 )
      {
LABEL_90:
        a2 = (__int64)src;
        result = sub_AC9970(*(_QWORD *)(*v4 + 8LL), (char *)src, (unsigned int)n);
        goto LABEL_66;
      }
      while ( 1 )
      {
        v74 = (_BYTE *)*v53;
        if ( *(_BYTE *)*v53 != 18 )
          goto LABEL_91;
        v54 = sub_C33340(p_src, a2, *v53, v10, v11);
        p_src = (void **)&v81;
        a2 = (__int64)(v74 + 24);
        if ( *((_QWORD *)v74 + 3) == v54 )
          sub_C3E660(&v81, a2);
        else
          sub_C3A850(&v81, a2);
        v75 = v82;
        if ( v82 > 0x40 )
        {
          p_src = (void **)&v81;
          v69 = sub_C444A0(&v81);
          v11 = -1;
          if ( v75 - v69 <= 0x40 )
            v11 = (__int64)*v81;
        }
        else
        {
          v11 = (__int64)v81;
        }
        v55 = (unsigned int)n;
        v10 = HIDWORD(n);
        v56 = (unsigned int)n + 1LL;
        if ( v56 > HIDWORD(n) )
        {
          p_src = &src;
          a2 = (__int64)&v85;
          v80 = v11;
          sub_C8D5F0(&src, &v85, v56, 8);
          v55 = (unsigned int)n;
          v11 = v80;
        }
        *((_QWORD *)src + v55) = v11;
        LODWORD(n) = n + 1;
        if ( v82 > 0x40 )
        {
          p_src = v81;
          if ( v81 )
            j_j___libc_free_0_0(v81);
        }
        if ( v5 == ++v53 )
          goto LABEL_90;
      }
    }
    return 0;
  }
  v65 = (_QWORD *)a2;
  src = &v85;
  for ( n = 0x1000000000LL; v5 != v65; ++v65 )
  {
    v76 = (_BYTE *)*v65;
    if ( *(_BYTE *)*v65 != 18 )
      goto LABEL_91;
    v66 = sub_C33340(p_src, a2, *v65, v10, v11);
    p_src = (void **)&v81;
    a2 = (__int64)(v76 + 24);
    if ( *((_QWORD *)v76 + 3) == v66 )
      sub_C3E660(&v81, a2);
    else
      sub_C3A850(&v81, a2);
    v77 = v82;
    if ( v82 > 0x40 )
    {
      p_src = (void **)&v81;
      v70 = sub_C444A0(&v81);
      v11 = 0xFFFFFFFFLL;
      if ( v77 - v70 <= 0x40 )
        v11 = *(unsigned int *)v81;
    }
    else
    {
      v11 = (unsigned int)v81;
    }
    v67 = (unsigned int)n;
    v10 = HIDWORD(n);
    v68 = (unsigned int)n + 1LL;
    if ( v68 > HIDWORD(n) )
    {
      p_src = &src;
      a2 = (__int64)&v85;
      v79 = v11;
      sub_C8D5F0(&src, &v85, v68, 4);
      v67 = (unsigned int)n;
      v11 = v79;
    }
    *((_DWORD *)src + v67) = v11;
    LODWORD(n) = n + 1;
    if ( v82 > 0x40 )
    {
      p_src = v81;
      if ( v81 )
        j_j___libc_free_0_0(v81);
    }
  }
  a2 = (__int64)src;
  result = sub_AC9940(*(_QWORD *)(*v4 + 8LL), (char *)src, (unsigned int)n);
LABEL_66:
  v25 = src;
  if ( src != &v85 )
  {
LABEL_32:
    v71 = result;
    _libc_free(v25, a2);
    return v71;
  }
  return result;
}
