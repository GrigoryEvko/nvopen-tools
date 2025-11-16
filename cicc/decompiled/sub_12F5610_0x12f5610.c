// Function: sub_12F5610
// Address: 0x12f5610
//
_QWORD *__fastcall sub_12F5610(__int64 a1, _BYTE **a2, char **a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // rax
  _QWORD **v6; // rbx
  _BYTE *v7; // r14
  __int64 v8; // rsi
  _QWORD *v9; // r15
  _QWORD *v10; // rbx
  int v11; // ecx
  _QWORD *v12; // r13
  int v13; // r14d
  char v14; // r12
  _QWORD *v15; // r12
  int v16; // ecx
  _QWORD *v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // r14
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rdx
  int v23; // eax
  char *v24; // r14
  size_t v25; // rax
  size_t v26; // r13
  char *v27; // rdx
  char *v28; // rdi
  char *v29; // rdx
  size_t v30; // rcx
  char *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  _QWORD *v34; // rbx
  _QWORD *v35; // r12
  _QWORD *v36; // r12
  __int64 v37; // r8
  __int64 v38; // r12
  __int64 v39; // rbx
  __int64 v40; // rdi
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  _QWORD *v45; // rbx
  __int64 v46; // r14
  char *v47; // r12
  size_t v48; // rax
  __int64 v49; // r12
  char *v50; // rdi
  size_t v51; // rdx
  _QWORD *v53; // [rsp+8h] [rbp-148h]
  __int64 v54; // [rsp+10h] [rbp-140h]
  _QWORD *v57; // [rsp+30h] [rbp-120h]
  _QWORD *v58; // [rsp+38h] [rbp-118h]
  int v59; // [rsp+38h] [rbp-118h]
  char *s; // [rsp+48h] [rbp-108h] BYREF
  _QWORD v61[4]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v62; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+78h] [rbp-D8h]
  __int64 v64; // [rsp+80h] [rbp-D0h]
  char *v65; // [rsp+90h] [rbp-C0h] BYREF
  size_t n; // [rsp+98h] [rbp-B8h]
  _QWORD src[2]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE v68[16]; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+C0h] [rbp-90h]
  _QWORD *v70; // [rsp+E0h] [rbp-70h]
  int v71; // [rsp+F0h] [rbp-60h]
  _QWORD *v72; // [rsp+100h] [rbp-50h]
  unsigned int v73; // [rsp+110h] [rbp-40h]

  v5 = *(_QWORD **)(a1 + 8);
  v6 = *(_QWORD ***)a1;
  v62 = 0;
  v63 = 0;
  v7 = *a2;
  v57 = v5;
  v8 = *(unsigned __int8 *)(a5 + 240);
  v64 = 0x1000000000LL;
  v9 = *v6;
  v53 = *v6;
  sub_16033C0(**v6, v8);
  if ( (*v7 & 1) != 0 )
    sub_12F5270((__int64)v9, (__int64)&v62);
  v10 = v6 + 1;
  sub_167C560(v68, v53);
  v11 = 1;
  v12 = v7;
  while ( 1 )
  {
    v59 = v11;
    if ( v57 == v10 )
      break;
    v15 = (_QWORD *)*v10++;
    sub_2240AE0(a4, v15 + 22);
    sub_16033C0(*v15, *(unsigned __int8 *)(a5 + 240));
    v16 = v59;
    if ( v59 == 63 )
    {
      v13 = 0;
      v58 = v12 + 1;
    }
    else
    {
      v58 = v12;
      v13 = v16 + 1;
    }
    if ( (*v12 & (1LL << v16)) != 0 )
      sub_12F5270((__int64)v15, (__int64)&v62);
    v61[0] = v15;
    src[0] = 0;
    v14 = sub_167DAB0(v68, v61, 0, &v65);
    if ( v61[0] )
    {
      v54 = v61[0];
      sub_1633490(v61[0]);
      j_j___libc_free_0(v54, 736);
    }
    if ( src[0] )
      ((void (__fastcall *)(char **, char **, __int64))src[0])(&v65, &v65, 3);
    if ( v14 )
    {
      v65 = 0;
      sub_1C3E9C0(&v65);
      v47 = v65;
      if ( v65 )
      {
        v48 = strlen(v65);
        if ( v48 > 0x3FFFFFFFFFFFFFFFLL - (__int64)a3[1] )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490(a3, v47, v48, a3);
        if ( v65 )
          j_j___libc_free_0_0(v65);
      }
      sub_1633490(v53);
      v33 = 736;
      j_j___libc_free_0(v53, 736);
      for ( ; v57 != v10; ++v10 )
      {
        v49 = *v10;
        if ( *v10 )
        {
          sub_1633490(*v10);
          v33 = 736;
          j_j___libc_free_0(v49, 736);
        }
      }
      v53 = 0;
      goto LABEL_41;
    }
    v12 = v58;
    v11 = v13;
  }
  if ( v53 + 3 != (_QWORD *)v53[4] )
  {
    v17 = (_QWORD *)v53[4];
    do
    {
      v19 = v17 - 7;
      if ( !v17 )
        v19 = 0;
      v20 = sub_1649960(v19);
      v21 = v62 + 8LL * (unsigned int)v63;
      v23 = sub_16D1B30(&v62, v20, v22);
      if ( v23 == -1 )
        v18 = v62 + 8LL * (unsigned int)v63;
      else
        v18 = v62 + 8LL * v23;
      if ( v21 != v18 )
        *((_WORD *)v19 + 16) = v19[4] & 0xBFC0 | 0x4007;
      v17 = (_QWORD *)v17[1];
    }
    while ( v53 + 3 != v17 );
  }
  s = 0;
  sub_1C3E9C0(&s);
  v24 = s;
  if ( s )
  {
    v65 = (char *)src;
    v25 = strlen(s);
    v61[0] = v25;
    v26 = v25;
    if ( v25 > 0xF )
    {
      v65 = (char *)sub_22409D0(&v65, v61, 0);
      v50 = v65;
      src[0] = v61[0];
    }
    else
    {
      if ( v25 == 1 )
      {
        LOBYTE(src[0]) = *v24;
        v27 = (char *)src;
        goto LABEL_30;
      }
      if ( !v25 )
      {
        v27 = (char *)src;
LABEL_30:
        n = v25;
        v27[v25] = 0;
        v28 = *a3;
        if ( v65 == (char *)src )
        {
          v51 = n;
          if ( n )
          {
            if ( n == 1 )
              *v28 = src[0];
            else
              memcpy(v28, src, n);
            v51 = n;
            v28 = *a3;
          }
          a3[1] = (char *)v51;
          v28[v51] = 0;
          v28 = v65;
          goto LABEL_34;
        }
        v29 = (char *)src[0];
        v30 = n;
        if ( v28 == (char *)(a3 + 2) )
        {
          *a3 = v65;
          a3[1] = (char *)v30;
          a3[2] = v29;
        }
        else
        {
          v31 = a3[2];
          *a3 = v65;
          a3[1] = (char *)v30;
          a3[2] = v29;
          if ( v28 )
          {
            v65 = v28;
            src[0] = v31;
LABEL_34:
            n = 0;
            *v28 = 0;
            if ( v65 != (char *)src )
              j_j___libc_free_0(v65, src[0] + 1LL);
            if ( s )
              j_j___libc_free_0_0(s);
            s = 0;
            goto LABEL_39;
          }
        }
        v65 = (char *)src;
        v28 = (char *)src;
        goto LABEL_34;
      }
      v50 = (char *)src;
    }
    memcpy(v50, v24, v26);
    v25 = v61[0];
    v27 = v65;
    goto LABEL_30;
  }
LABEL_39:
  sub_1611EE0(v61);
  v32 = sub_1CB9110(a5 + 208);
  sub_1619140(v61, v32, 1);
  v33 = (__int64)v53;
  if ( (unsigned __int8)sub_1619BD0(v61, v53) && !LOBYTE(qword_4F96820[20]) )
  {
    sub_1611EE0(&v65);
    v42 = sub_1CC63C0();
    sub_1619140(&v65, v42, 0);
    v33 = (__int64)v53;
    sub_1619BD0(&v65, v53);
    sub_160FE50(&v65);
  }
  sub_160FE50(v61);
LABEL_41:
  if ( v73 )
  {
    v34 = v72;
    v35 = &v72[2 * v73];
    do
    {
      if ( *v34 != -4 && *v34 != -8 )
      {
        v33 = v34[1];
        if ( v33 )
          sub_161E7C0(v34 + 1);
      }
      v34 += 2;
    }
    while ( v35 != v34 );
  }
  j___libc_free_0(v72);
  if ( !v71 )
    goto LABEL_49;
  v43 = sub_16704E0();
  v44 = sub_16704F0();
  v45 = v70;
  v46 = v44;
  v36 = &v70[v71];
  if ( v70 != v36 )
  {
    do
    {
      v33 = v43;
      if ( !(unsigned __int8)sub_1670560(*v45, v43) )
      {
        v33 = v46;
        sub_1670560(*v45, v46);
      }
      ++v45;
    }
    while ( v36 != v45 );
LABEL_49:
    v36 = v70;
  }
  j___libc_free_0(v36);
  j___libc_free_0(v69);
  if ( HIDWORD(v63) )
  {
    v37 = v62;
    if ( (_DWORD)v63 )
    {
      v38 = 8LL * (unsigned int)v63;
      v39 = 0;
      do
      {
        v40 = *(_QWORD *)(v37 + v39);
        if ( v40 != -8 && v40 )
        {
          _libc_free(v40, v33);
          v37 = v62;
        }
        v39 += 8;
      }
      while ( v39 != v38 );
    }
  }
  else
  {
    v37 = v62;
  }
  _libc_free(v37, v33);
  return v53;
}
