// Function: sub_15A1590
// Address: 0x15a1590
//
__int64 __fastcall sub_15A1590(size_t n, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r14d
  char v6; // al
  __int64 *v7; // rsi
  __int64 v8; // r13
  size_t v9; // r15
  char *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  void *v13; // rdi
  __int64 v14; // r12
  char v16; // al
  _QWORD *v17; // r15
  __int64 v18; // rdx
  char *v19; // r13
  char *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rdi
  char v24; // al
  _QWORD *v25; // r15
  __int64 v26; // rdx
  char *v27; // r13
  char *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rsi
  unsigned int v33; // ebx
  __int16 v34; // r13
  _WORD *v35; // rax
  _WORD *v36; // rdx
  char *v37; // r14
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // rsi
  unsigned int v43; // ebx
  __int64 v44; // r13
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  char *v47; // r14
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  char *v52; // rax
  char *v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rsi
  unsigned int v58; // ebx
  int v59; // r13d
  _DWORD *v60; // rax
  _DWORD *v61; // rdx
  char *v62; // r14
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // [rsp+8h] [rbp-D8h]
  __int64 v67; // [rsp+8h] [rbp-D8h]
  __int64 v68; // [rsp+8h] [rbp-D8h]
  __int16 *v69; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v70; // [rsp+18h] [rbp-C8h]
  void *s; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v72; // [rsp+28h] [rbp-B8h]
  _BYTE v73[176]; // [rsp+30h] [rbp-B0h] BYREF

  v4 = n;
  v6 = *((_BYTE *)a2 + 16);
  if ( v6 == 13 )
  {
    if ( !(unsigned __int8)sub_1642F90(*a2, 8) )
    {
      if ( (unsigned __int8)sub_1642F90(*a2, 16) )
      {
        v17 = (_QWORD *)a2[3];
        if ( *((_DWORD *)a2 + 8) > 0x40u )
          v17 = (_QWORD *)*v17;
        v18 = (unsigned int)n;
        s = v73;
        v19 = v73;
        v72 = 0x1000000000LL;
        if ( (unsigned int)n > 0x10 )
        {
          sub_16CD150(&s, v73, (unsigned int)n, 2);
          v19 = (char *)s;
          v18 = (unsigned int)n;
        }
        v20 = &v19[2 * v18];
        LODWORD(v72) = n;
        if ( v20 != v19 )
        {
          do
          {
            *(_WORD *)v19 = (_WORD)v17;
            v19 += 2;
          }
          while ( v20 != v19 );
          v19 = (char *)s;
          v18 = (unsigned int)v72;
        }
        v66 = v18;
        v21 = sub_16498A0(a2);
        v22 = sub_1599550(v21, v19, v66);
        v13 = s;
        v14 = v22;
        if ( s == v73 )
          return v14;
      }
      else
      {
        v24 = sub_1642F90(*a2, 32);
        v25 = (_QWORD *)a2[3];
        if ( v24 )
        {
          if ( *((_DWORD *)a2 + 8) > 0x40u )
            v25 = (_QWORD *)*v25;
          v26 = (unsigned int)n;
          s = v73;
          v27 = v73;
          v72 = 0x1000000000LL;
          if ( (unsigned int)n > 0x10 )
          {
            sub_16CD150(&s, v73, (unsigned int)n, 4);
            v27 = (char *)s;
            v26 = (unsigned int)n;
          }
          v28 = &v27[4 * v26];
          LODWORD(v72) = n;
          if ( v28 != v27 )
          {
            do
            {
              *(_DWORD *)v27 = (_DWORD)v25;
              v27 += 4;
            }
            while ( v28 != v27 );
            v27 = (char *)s;
            v26 = (unsigned int)v72;
          }
          v67 = v26;
          v29 = sub_16498A0(a2);
          v30 = sub_1599580(v29, v27, v67);
          v13 = s;
          v14 = v30;
          if ( s == v73 )
            return v14;
        }
        else
        {
          if ( *((_DWORD *)a2 + 8) > 0x40u )
            v25 = (_QWORD *)*v25;
          v51 = (unsigned int)n;
          v72 = 0x1000000000LL;
          v52 = v73;
          s = v73;
          if ( (unsigned int)n > 0x10 )
          {
            sub_16CD150(&s, v73, (unsigned int)n, 8);
            v52 = (char *)s;
            v51 = (unsigned int)n;
          }
          v53 = &v52[8 * v51];
          LODWORD(v72) = n;
          if ( v52 != v53 )
          {
            do
            {
              *(_QWORD *)v52 = v25;
              v52 += 8;
            }
            while ( v53 != v52 );
            v53 = (char *)s;
            v51 = (unsigned int)v72;
          }
          v68 = v51;
          v54 = sub_16498A0(a2);
          v55 = sub_15995C0(v54, v53, v68);
          v13 = s;
          v14 = v55;
          if ( s == v73 )
            return v14;
        }
      }
      goto LABEL_9;
    }
    v7 = (__int64 *)a2[3];
    v8 = a2[3];
    if ( *((_DWORD *)a2 + 8) > 0x40u )
      v8 = *v7;
    v9 = (unsigned int)n;
    s = v73;
    v72 = 0x1000000000LL;
    if ( (unsigned int)n > 0x10 )
    {
      sub_16CD150(&s, v73, (unsigned int)n, 1);
      LODWORD(v72) = n;
      v23 = s;
    }
    else
    {
      LODWORD(v72) = n;
      if ( !(_DWORD)n )
      {
        v10 = v73;
        goto LABEL_8;
      }
      v23 = v73;
    }
    memset(v23, (unsigned __int8)v8, v4);
    v10 = (char *)s;
    v9 = (unsigned int)v72;
LABEL_8:
    v11 = sub_16498A0(a2);
    v12 = sub_1599510(v11, v10, v9);
    v13 = s;
    v14 = v12;
    if ( s == v73 )
      return v14;
LABEL_9:
    _libc_free((unsigned __int64)v13);
    return v14;
  }
  if ( v6 == 14 )
  {
    v16 = *(_BYTE *)(*a2 + 8LL);
    switch ( v16 )
    {
      case 1:
        v31 = sub_16982C0(n, a2, a3, a4);
        v32 = a2 + 4;
        if ( a2[4] == v31 )
          sub_169D930(&v69, v32);
        else
          sub_169D7E0(&v69, v32);
        v33 = v70;
        if ( v70 > 0x40 )
        {
          v34 = -1;
          if ( v33 - (unsigned int)sub_16A57B0(&v69) <= 0x40 )
            v34 = *v69;
        }
        else
        {
          v34 = (__int16)v69;
        }
        v72 = 0x1000000000LL;
        v35 = v73;
        s = v73;
        if ( (unsigned int)n > 0x10 )
        {
          sub_16CD150(&s, v73, (unsigned int)n, 2);
          v35 = s;
        }
        v36 = &v35[(unsigned int)n];
        for ( LODWORD(v72) = n; v36 != v35; ++v35 )
          *v35 = v34;
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        v37 = (char *)s;
        v38 = (unsigned int)v72;
        v39 = sub_16498A0(a2);
        v40 = sub_1599600(v39, v37, v38);
        v13 = s;
        v14 = v40;
        if ( s == v73 )
          return v14;
        goto LABEL_9;
      case 2:
        v56 = sub_16982C0(n, a2, a3, a4);
        v57 = a2 + 4;
        if ( a2[4] == v56 )
          sub_169D930(&v69, v57);
        else
          sub_169D7E0(&v69, v57);
        v58 = v70;
        if ( v70 > 0x40 )
        {
          v59 = -1;
          if ( v58 - (unsigned int)sub_16A57B0(&v69) <= 0x40 )
            v59 = *(_DWORD *)v69;
        }
        else
        {
          v59 = (int)v69;
        }
        v72 = 0x1000000000LL;
        v60 = v73;
        s = v73;
        if ( (unsigned int)n > 0x10 )
        {
          sub_16CD150(&s, v73, (unsigned int)n, 4);
          v60 = s;
        }
        v61 = &v60[(unsigned int)n];
        for ( LODWORD(v72) = n; v61 != v60; ++v60 )
          *v60 = v59;
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        v62 = (char *)s;
        v63 = (unsigned int)v72;
        v64 = sub_16498A0(a2);
        v65 = sub_1599630(v64, v62, v63);
        v13 = s;
        v14 = v65;
        if ( s == v73 )
          return v14;
        goto LABEL_9;
      case 3:
        v41 = sub_16982C0(n, a2, a3, a4);
        v42 = a2 + 4;
        if ( a2[4] == v41 )
          sub_169D930(&v69, v42);
        else
          sub_169D7E0(&v69, v42);
        v43 = v70;
        if ( v70 > 0x40 )
        {
          v44 = -1;
          if ( v43 - (unsigned int)sub_16A57B0(&v69) <= 0x40 )
            v44 = *(_QWORD *)v69;
        }
        else
        {
          v44 = (__int64)v69;
        }
        v72 = 0x1000000000LL;
        v45 = v73;
        s = v73;
        if ( (unsigned int)n > 0x10 )
        {
          sub_16CD150(&s, v73, (unsigned int)n, 8);
          v45 = s;
        }
        v46 = &v45[(unsigned int)n];
        for ( LODWORD(v72) = n; v46 != v45; ++v45 )
          *v45 = v44;
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        v47 = (char *)s;
        v48 = (unsigned int)v72;
        v49 = sub_16498A0(a2);
        v50 = sub_1599670(v49, v47, v48);
        v13 = s;
        v14 = v50;
        if ( s == v73 )
          return v14;
        goto LABEL_9;
    }
  }
  return sub_15A0390((unsigned int)n, (__int64)a2);
}
