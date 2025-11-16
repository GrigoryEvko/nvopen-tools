// Function: sub_C84CF0
// Address: 0xc84cf0
//
__int64 __fastcall sub_C84CF0(unsigned __int8 **a1, unsigned __int8 a2, unsigned int a3)
{
  unsigned int v3; // r13d
  unsigned __int64 p_s2; // rsi
  unsigned __int8 *v5; // rdi
  unsigned __int64 v6; // rdx
  size_t v7; // r14
  unsigned __int64 v8; // rax
  const char *v9; // rbx
  char v10; // al
  unsigned int v11; // r12d
  char v12; // r13
  size_t v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rcx
  unsigned __int8 *v16; // r8
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned __int8 *v19; // rdx
  unsigned __int8 v20; // di
  unsigned __int8 *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  size_t v24; // r15
  _BYTE *v25; // r9
  int v26; // eax
  __int64 v28; // rdx
  unsigned __int8 **v29; // rdx
  char *v30; // rdi
  size_t v31; // r12
  const void *v32; // r14
  size_t v33; // rax
  _BYTE *v34; // r14
  __int64 v35; // r12
  bool v36; // zf
  size_t v37; // rdi
  char v38; // r13
  size_t v39; // rdi
  const void *v40; // r9
  size_t v41; // rbx
  unsigned __int8 **v42; // rdx
  unsigned __int64 v43; // rcx
  unsigned __int8 *v44; // [rsp+8h] [rbp-2A8h]
  unsigned __int64 v45; // [rsp+10h] [rbp-2A0h]
  unsigned __int8 *v46; // [rsp+10h] [rbp-2A0h]
  unsigned int v47; // [rsp+1Ch] [rbp-294h]
  void *s1; // [rsp+28h] [rbp-288h]
  _BYTE *v51; // [rsp+38h] [rbp-278h]
  const void *v52; // [rsp+38h] [rbp-278h]
  const void *v53; // [rsp+38h] [rbp-278h]
  unsigned __int8 *v54; // [rsp+40h] [rbp-270h] BYREF
  unsigned __int64 v55; // [rsp+48h] [rbp-268h]
  _BYTE *v56; // [rsp+50h] [rbp-260h] BYREF
  __int64 v57; // [rsp+58h] [rbp-258h]
  _BYTE v58[256]; // [rsp+60h] [rbp-250h] BYREF
  void *s2; // [rsp+160h] [rbp-150h] BYREF
  size_t v60; // [rsp+168h] [rbp-148h]
  unsigned __int64 v61; // [rsp+170h] [rbp-140h]
  _BYTE v62[312]; // [rsp+178h] [rbp-138h] BYREF

  v3 = a3;
  if ( !a3 )
    v3 = 1;
  p_s2 = (unsigned __int64)a1[1];
  v5 = *a1;
  v56 = v58;
  v54 = v5;
  v55 = p_s2;
  v57 = 0x1000000000LL;
  s1 = (void *)sub_C80E20(v5, p_s2, v3);
  v7 = v6;
  if ( !v6 )
  {
    if ( !v55 )
    {
      v11 = 0;
      v60 = 0;
      s2 = v62;
      v61 = 256;
      goto LABEL_27;
    }
    goto LABEL_6;
  }
  v8 = v55;
  if ( v55 < v6 )
  {
    v55 = 0;
    v54 += v8;
LABEL_55:
    v11 = 0;
    v60 = 0;
    s2 = v62;
    v61 = 256;
    if ( v6 <= 0x100 )
      goto LABEL_75;
    goto LABEL_56;
  }
  v54 += v6;
  v55 -= v6;
  if ( !v55 )
    goto LABEL_55;
LABEL_6:
  v9 = "/";
  if ( v3 != 1 )
    v9 = "\\/";
  v10 = 47;
  v47 = v3;
  if ( v3 == 3 )
    v10 = 92;
  v11 = 0;
  v12 = v10;
  do
  {
    while ( 1 )
    {
      v13 = strlen(v9);
      p_s2 = (unsigned __int64)v9;
      v14 = sub_C934D0(&v54, v9, v13, 0);
      if ( v14 == -1 )
      {
        v17 = v55;
        v14 = v55;
      }
      else
      {
        v15 = v55;
        v16 = v54;
        v17 = v14;
        if ( v14 < v55 )
          goto LABEL_13;
        v14 = v55;
      }
      v16 = v54;
      v15 = v14;
LABEL_13:
      if ( v15 < v17 )
      {
        v55 = 0;
        v54 = &v16[v15];
      }
      else
      {
        v18 = v15 - v17;
        v19 = &v16[v17];
        v54 = v19;
        if ( v18 == -1 )
        {
          v55 = -1;
          v20 = *v19;
          p_s2 = -2;
LABEL_16:
          v21 = v19 + 1;
          v55 = p_s2;
          v54 = v21;
          LOBYTE(v21) = v12 != (char)v20;
          v11 |= (unsigned int)v21;
          goto LABEL_17;
        }
        v55 = v18;
        if ( v18 )
        {
          p_s2 = v18 - 1;
          v43 = v18 - 1;
          v20 = *v19;
          LOBYTE(v43) = v43 == 0;
          v11 |= v43;
          goto LABEL_16;
        }
      }
LABEL_17:
      if ( !v14 )
        goto LABEL_46;
      if ( v14 == 1 )
      {
        if ( *v16 == 46 )
        {
LABEL_46:
          v11 = 1;
          goto LABEL_43;
        }
        break;
      }
      if ( !a2 )
        break;
      v22 = (unsigned int)v57;
      if ( v14 != 2 || *(_WORD *)v16 != 11822 )
        break;
      if ( !(_DWORD)v57
        || (v23 = (__int64)&v56[16 * (unsigned int)v57 - 16], *(_QWORD *)(v23 + 8) == 2) && **(_WORD **)v23 == 11822 )
      {
        if ( !v7 )
        {
          if ( (unsigned __int64)(unsigned int)v57 + 1 > HIDWORD(v57) )
          {
            p_s2 = (unsigned __int64)v58;
            v46 = v16;
            sub_C8D5F0(&v56, v58, (unsigned int)v57 + 1LL, 16);
            v22 = (unsigned int)v57;
            v16 = v46;
          }
          v42 = (unsigned __int8 **)&v56[16 * v22];
          *v42 = v16;
          v42[1] = (unsigned __int8 *)2;
          LODWORD(v57) = v57 + 1;
        }
        v11 = a2;
        goto LABEL_43;
      }
      v11 = a2;
      LODWORD(v57) = v57 - 1;
      if ( !v55 )
        goto LABEL_25;
    }
    v28 = (unsigned int)v57;
    if ( (unsigned __int64)(unsigned int)v57 + 1 > HIDWORD(v57) )
    {
      p_s2 = (unsigned __int64)v58;
      v44 = v16;
      v45 = v14;
      sub_C8D5F0(&v56, v58, (unsigned int)v57 + 1LL, 16);
      v28 = (unsigned int)v57;
      v16 = v44;
      v14 = v45;
    }
    v29 = (unsigned __int8 **)&v56[16 * v28];
    *v29 = v16;
    v29[1] = (unsigned __int8 *)v14;
    LODWORD(v57) = v57 + 1;
LABEL_43:
    ;
  }
  while ( v55 );
LABEL_25:
  v3 = v47;
  v60 = 0;
  s2 = v62;
  v61 = 256;
  if ( v7 > 0x100 )
  {
LABEL_56:
    sub_C8D290(&s2, v62, v7, 1);
    v30 = (char *)s2 + v60;
    goto LABEL_57;
  }
  if ( !v7 )
    goto LABEL_27;
LABEL_75:
  v30 = v62;
LABEL_57:
  p_s2 = (unsigned __int64)s1;
  memcpy(v30, s1, v7);
LABEL_27:
  v24 = v7 + v60;
  v60 += v7;
  if ( v3 != 1 )
  {
    p_s2 = v3;
    sub_C83970((char **)&s2, v3);
    v24 = v60;
  }
  v25 = s2;
  if ( v24 != v7 )
    goto LABEL_58;
  if ( v24 )
  {
    p_s2 = (unsigned __int64)s2;
    v51 = s2;
    v26 = memcmp(s1, s2, v24);
    v25 = v51;
    LOBYTE(v26) = v26 != 0;
    v11 |= v26;
  }
  if ( (_BYTE)v11 )
  {
LABEL_58:
    if ( (_DWORD)v57 )
    {
      v31 = *((_QWORD *)v56 + 1);
      v32 = *(const void **)v56;
      if ( v31 + v24 > v61 )
      {
        sub_C8D290(&s2, v62, v31 + v24, 1);
        v25 = s2;
        v24 = v60;
      }
      if ( v31 )
      {
        memcpy(&v25[v24], v32, v31);
        v24 = v60;
      }
      v33 = v31 + v24;
      v60 = v31 + v24;
      v34 = v56 + 16;
      v35 = (__int64)&v56[16 * (unsigned int)v57];
      if ( (_BYTE *)v35 != v56 + 16 )
      {
        v36 = v3 == 3;
        v37 = v33;
        v38 = 92;
        if ( !v36 )
          v38 = 47;
        do
        {
          v40 = *(const void **)v34;
          v41 = *((_QWORD *)v34 + 1);
          if ( v37 + 1 > v61 )
          {
            v53 = *(const void **)v34;
            sub_C8D290(&s2, v62, v37 + 1, 1);
            v37 = v60;
            v40 = v53;
          }
          *((_BYTE *)s2 + v37) = v38;
          v39 = v60 + 1;
          v60 = v39;
          if ( v41 + v39 > v61 )
          {
            v52 = v40;
            sub_C8D290(&s2, v62, v41 + v39, 1);
            v39 = v60;
            v40 = v52;
          }
          if ( v41 )
          {
            memcpy((char *)s2 + v39, v40, v41);
            v39 = v60;
          }
          v37 = v41 + v39;
          v34 += 16;
          v60 = v37;
        }
        while ( (_BYTE *)v35 != v34 );
      }
    }
    p_s2 = (unsigned __int64)&s2;
    v11 = 1;
    sub_C844F0(a1, &s2);
    v25 = s2;
  }
  if ( v25 != v62 )
    _libc_free(v25, p_s2);
  if ( v56 != v58 )
    _libc_free(v56, p_s2);
  return v11;
}
