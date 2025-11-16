// Function: sub_159A200
// Address: 0x159a200
//
__int64 __fastcall sub_159A200(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rbx
  __int64 *v5; // r13
  __int64 *v7; // r12
  __int64 *v8; // rax
  __int16 **p_src; // rdi
  __int64 v10; // rcx
  bool v11; // r8
  __int64 result; // rax
  char v13; // al
  __int64 *v14; // r15
  __int64 *v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  size_t v20; // r12
  char *v21; // r13
  __int64 v22; // rax
  __int64 **v23; // rax
  __int64 *v24; // rax
  char v25; // al
  __int64 *v26; // r14
  __int64 *v27; // r12
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rax
  char *v31; // r14
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 *v34; // r15
  __int64 *v35; // r12
  __int64 v36; // rax
  _QWORD *v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r12
  char *v41; // r14
  size_t v42; // r13
  __int64 v43; // rdi
  __int64 *v44; // r15
  __int64 *v45; // r14
  __int64 v46; // rax
  _QWORD *v47; // r12
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 **v50; // rax
  __int64 *v51; // r14
  __int64 *v52; // r12
  __int64 v53; // rax
  __int16 v54; // r8
  __int64 v55; // rax
  char *v56; // r14
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 *v59; // r15
  __int64 *v60; // r12
  __int64 v61; // rax
  _QWORD *v62; // r14
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 *v65; // r14
  __int64 *v66; // r12
  __int64 v67; // rax
  int v68; // r8d
  __int64 v69; // rax
  char *v70; // r14
  __int64 v71; // r12
  __int64 v72; // rax
  int v73; // eax
  int v74; // eax
  int v75; // eax
  __int64 v76; // [rsp+8h] [rbp-D8h]
  __int64 v77; // [rsp+8h] [rbp-D8h]
  unsigned int v78; // [rsp+8h] [rbp-D8h]
  __int64 v79; // [rsp+8h] [rbp-D8h]
  unsigned int v80; // [rsp+8h] [rbp-D8h]
  __int64 v81; // [rsp+8h] [rbp-D8h]
  unsigned int v82; // [rsp+8h] [rbp-D8h]
  __int64 v83; // [rsp+8h] [rbp-D8h]
  __int16 v84; // [rsp+8h] [rbp-D8h]
  int v85; // [rsp+8h] [rbp-D8h]
  __int16 *v86; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v87; // [rsp+18h] [rbp-C8h]
  void *src; // [rsp+20h] [rbp-C0h] BYREF
  size_t n; // [rsp+28h] [rbp-B8h]
  _BYTE v90[176]; // [rsp+30h] [rbp-B0h] BYREF

  if ( !a3 )
    return sub_1598F00(a1);
  v4 = (__int64 *)*a2;
  v5 = a2;
  if ( *(_BYTE *)(*a2 + 16) != 9 )
  {
    if ( !sub_1593BB0(*a2, (__int64)a2, a3, a4) )
      goto LABEL_8;
    v7 = &a2[a3];
    if ( v7 != a2 )
    {
LABEL_22:
      v24 = a2;
      while ( v4 == (__int64 *)*v24 )
      {
        if ( ++v24 == v7 )
          return sub_1598F00(a1);
      }
LABEL_8:
      p_src = (__int16 **)*v4;
      v11 = sub_15958A0(*v4);
      result = 0;
      if ( !v11 )
        return result;
      v13 = *((_BYTE *)v4 + 16);
      if ( v13 == 13 )
      {
        if ( (unsigned __int8)sub_1642F90(*v4, 8) )
        {
          v14 = &a2[a3];
          v15 = a2;
          src = v90;
          n = 0x1000000000LL;
          if ( a2 == v14 )
          {
LABEL_18:
            v19 = sub_16498A0(*a2);
            v20 = (unsigned int)n;
            v21 = (char *)src;
            v22 = sub_1644C60(v19, 8);
            v23 = (__int64 **)sub_1645D80(v22, v20);
            result = sub_15991C0(v21, v20, v23);
            goto LABEL_27;
          }
          while ( 1 )
          {
            v16 = *v15;
            if ( *(_BYTE *)(*v15 + 16) != 13 )
              goto LABEL_26;
            v17 = *(_QWORD **)(v16 + 24);
            if ( *(_DWORD *)(v16 + 32) > 0x40u )
              v17 = (_QWORD *)*v17;
            v18 = (unsigned int)n;
            if ( (unsigned int)n >= HIDWORD(n) )
            {
              sub_16CD150(&src, v90, 0, 1);
              v18 = (unsigned int)n;
            }
            ++v15;
            *((_BYTE *)src + v18) = (_BYTE)v17;
            LODWORD(n) = n + 1;
            if ( v14 == v15 )
              goto LABEL_18;
          }
        }
        if ( (unsigned __int8)sub_1642F90(*v4, 16) )
        {
          v44 = &a2[a3];
          v45 = a2;
          src = v90;
          for ( n = 0x1000000000LL; v44 != v45; LODWORD(n) = n + 1 )
          {
            v46 = *v45;
            if ( *(_BYTE *)(*v45 + 16) != 13 )
              goto LABEL_26;
            v47 = *(_QWORD **)(v46 + 24);
            if ( *(_DWORD *)(v46 + 32) > 0x40u )
              v47 = (_QWORD *)*v47;
            v48 = (unsigned int)n;
            if ( (unsigned int)n >= HIDWORD(n) )
            {
              sub_16CD150(&src, v90, 0, 2);
              v48 = (unsigned int)n;
            }
            ++v45;
            *((_WORD *)src + v48) = (_WORD)v47;
          }
          v49 = sub_16498A0(*a2);
          v40 = (unsigned int)n;
          v41 = (char *)src;
          v42 = 2LL * (unsigned int)n;
          v43 = sub_1644C60(v49, 16);
LABEL_67:
          v50 = (__int64 **)sub_1645D80(v43, v40);
          result = sub_15991C0(v41, v42, v50);
          goto LABEL_27;
        }
        if ( (unsigned __int8)sub_1642F90(*v4, 32) )
        {
          v34 = &a2[a3];
          v35 = a2;
          src = v90;
          n = 0x1000000000LL;
          if ( a2 != v34 )
          {
            while ( 1 )
            {
              v36 = *v35;
              if ( *(_BYTE *)(*v35 + 16) != 13 )
                break;
              v37 = *(_QWORD **)(v36 + 24);
              if ( *(_DWORD *)(v36 + 32) > 0x40u )
                v37 = (_QWORD *)*v37;
              v38 = (unsigned int)n;
              if ( (unsigned int)n >= HIDWORD(n) )
              {
                sub_16CD150(&src, v90, 0, 4);
                v38 = (unsigned int)n;
              }
              ++v35;
              *((_DWORD *)src + v38) = (_DWORD)v37;
              LODWORD(n) = n + 1;
              if ( v34 == v35 )
                goto LABEL_58;
            }
LABEL_26:
            result = 0;
            goto LABEL_27;
          }
LABEL_58:
          v39 = sub_16498A0(*a2);
          v40 = (unsigned int)n;
          v41 = (char *)src;
          v42 = 4LL * (unsigned int)n;
          v43 = sub_1644C60(v39, 32);
          goto LABEL_67;
        }
        if ( (unsigned __int8)sub_1642F90(*v4, 64) )
        {
          v59 = &a2[a3];
          v60 = a2;
          src = v90;
          for ( n = 0x1000000000LL; v59 != v60; LODWORD(n) = n + 1 )
          {
            v61 = *v60;
            if ( *(_BYTE *)(*v60 + 16) != 13 )
              goto LABEL_26;
            v62 = *(_QWORD **)(v61 + 24);
            if ( *(_DWORD *)(v61 + 32) > 0x40u )
              v62 = (_QWORD *)*v62;
            v63 = (unsigned int)n;
            if ( (unsigned int)n >= HIDWORD(n) )
            {
              sub_16CD150(&src, v90, 0, 8);
              v63 = (unsigned int)n;
            }
            ++v60;
            *((_QWORD *)src + v63) = v62;
          }
          v64 = sub_16498A0(*a2);
          v40 = (unsigned int)n;
          v41 = (char *)src;
          v42 = 8LL * (unsigned int)n;
          v43 = sub_1644C60(v64, 64);
          goto LABEL_67;
        }
      }
      else if ( v13 == 14 )
      {
        v25 = *(_BYTE *)(*v4 + 8);
        switch ( v25 )
        {
          case 1:
            v51 = &a2[a3];
            v52 = a2;
            src = v90;
            for ( n = 0x1000000000LL; v51 != v52; ++v52 )
            {
              v79 = *v52;
              if ( *(_BYTE *)(*v52 + 16) != 14 )
                goto LABEL_26;
              v53 = sub_16982C0(p_src, a2, *v52, v10);
              p_src = &v86;
              a2 = (__int64 *)(v79 + 32);
              if ( *(_QWORD *)(v79 + 32) == v53 )
                sub_169D930(&v86, a2);
              else
                sub_169D7E0(&v86, a2);
              v80 = v87;
              if ( v87 > 0x40 )
              {
                p_src = &v86;
                v74 = sub_16A57B0(&v86);
                v54 = -1;
                if ( v80 - v74 <= 0x40 )
                  v54 = *v86;
              }
              else
              {
                v54 = (__int16)v86;
              }
              v55 = (unsigned int)n;
              if ( (unsigned int)n >= HIDWORD(n) )
              {
                p_src = (__int16 **)&src;
                a2 = (__int64 *)v90;
                v84 = v54;
                sub_16CD150(&src, v90, 0, 2);
                v55 = (unsigned int)n;
                v54 = v84;
              }
              *((_WORD *)src + v55) = v54;
              LODWORD(n) = n + 1;
              if ( v87 > 0x40 )
              {
                p_src = (__int16 **)v86;
                if ( v86 )
                  j_j___libc_free_0_0(v86);
              }
            }
            v56 = (char *)src;
            v57 = (unsigned int)n;
            v58 = sub_16498A0(*v5);
            result = sub_1599460(v58, v56, v57);
            break;
          case 2:
            v65 = &a2[a3];
            v66 = a2;
            src = v90;
            for ( n = 0x1000000000LL; v65 != v66; ++v66 )
            {
              v81 = *v66;
              if ( *(_BYTE *)(*v66 + 16) != 14 )
                goto LABEL_26;
              v67 = sub_16982C0(p_src, a2, *v66, v10);
              p_src = &v86;
              a2 = (__int64 *)(v81 + 32);
              if ( *(_QWORD *)(v81 + 32) == v67 )
                sub_169D930(&v86, a2);
              else
                sub_169D7E0(&v86, a2);
              v82 = v87;
              if ( v87 > 0x40 )
              {
                p_src = &v86;
                v75 = sub_16A57B0(&v86);
                v68 = -1;
                if ( v82 - v75 <= 0x40 )
                  v68 = *(_DWORD *)v86;
              }
              else
              {
                v68 = (int)v86;
              }
              v69 = (unsigned int)n;
              if ( (unsigned int)n >= HIDWORD(n) )
              {
                p_src = (__int16 **)&src;
                a2 = (__int64 *)v90;
                v85 = v68;
                sub_16CD150(&src, v90, 0, 4);
                v69 = (unsigned int)n;
                v68 = v85;
              }
              *((_DWORD *)src + v69) = v68;
              LODWORD(n) = n + 1;
              if ( v87 > 0x40 )
              {
                p_src = (__int16 **)v86;
                if ( v86 )
                  j_j___libc_free_0_0(v86);
              }
            }
            v70 = (char *)src;
            v71 = (unsigned int)n;
            v72 = sub_16498A0(*v5);
            result = sub_1599490(v72, v70, v71);
            break;
          case 3:
            v26 = &a2[a3];
            v27 = a2;
            src = v90;
            for ( n = 0x1000000000LL; v26 != v27; ++v27 )
            {
              v77 = *v27;
              if ( *(_BYTE *)(*v27 + 16) != 14 )
                goto LABEL_26;
              v28 = sub_16982C0(p_src, a2, *v27, v10);
              p_src = &v86;
              a2 = (__int64 *)(v77 + 32);
              if ( *(_QWORD *)(v77 + 32) == v28 )
                sub_169D930(&v86, a2);
              else
                sub_169D7E0(&v86, a2);
              v78 = v87;
              if ( v87 > 0x40 )
              {
                p_src = &v86;
                v73 = sub_16A57B0(&v86);
                v29 = -1;
                if ( v78 - v73 <= 0x40 )
                  v29 = *(_QWORD *)v86;
              }
              else
              {
                v29 = (__int64)v86;
              }
              v30 = (unsigned int)n;
              if ( (unsigned int)n >= HIDWORD(n) )
              {
                p_src = (__int16 **)&src;
                a2 = (__int64 *)v90;
                v83 = v29;
                sub_16CD150(&src, v90, 0, 8);
                v30 = (unsigned int)n;
                v29 = v83;
              }
              *((_QWORD *)src + v30) = v29;
              LODWORD(n) = n + 1;
              if ( v87 > 0x40 )
              {
                p_src = (__int16 **)v86;
                if ( v86 )
                  j_j___libc_free_0_0(v86);
              }
            }
            v31 = (char *)src;
            v32 = (unsigned int)n;
            v33 = sub_16498A0(*v5);
            result = sub_15994D0(v33, v31, v32);
            break;
          default:
            return 0;
        }
LABEL_27:
        if ( src != v90 )
        {
          v76 = result;
          _libc_free((unsigned __int64)src);
          return v76;
        }
        return result;
      }
      return 0;
    }
    return sub_1598F00(a1);
  }
  v7 = &a2[a3];
  if ( v7 != a2 )
  {
    v8 = a2;
    while ( ++v8 != v7 )
    {
      if ( v4 != (__int64 *)*v8 )
      {
        if ( sub_1593BB0(*a2, (__int64)a2, a3, a4) )
          goto LABEL_22;
        goto LABEL_8;
      }
    }
  }
  return sub_1599EF0(a1);
}
