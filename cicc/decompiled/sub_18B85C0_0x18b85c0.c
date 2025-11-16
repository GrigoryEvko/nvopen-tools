// Function: sub_18B85C0
// Address: 0x18b85c0
//
__int64 __fastcall sub_18B85C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // r12
  unsigned int v9; // ebx
  __int64 v11; // rbx
  __int64 i; // r15
  size_t v13; // rdx
  __int64 v14; // r15
  size_t v15; // rdx
  const void *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  const char *v19; // r8
  size_t v20; // r12
  size_t v21; // rax
  __int64 (__fastcall **v22)(__int64 *); // rdx
  _BYTE *v23; // rdi
  __int64 (__fastcall **v24)(__int64 *); // rax
  __int64 (__fastcall *v25)(__int64 *); // rsi
  size_t v26; // rdi
  __int64 (__fastcall *v27)(__int64 *); // rcx
  size_t v28; // rdx
  __int64 v29; // rax
  __int64 (__fastcall **v30)(__int64 *); // rdi
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 *v33; // r14
  __int64 (__fastcall ***v34)(__int64 *); // r15
  const char **v35; // rdi
  __int64 (__fastcall *v36)(__int64); // rax
  __int64 v37; // rax
  __int64 (__fastcall **v38)(__int64 *); // rdi
  __int64 *v39; // r14
  __int64 (__fastcall ***v40)(__int64 *); // r15
  const char **v41; // rdi
  __int64 (__fastcall *v42)(__int64); // rax
  __int64 (__fastcall **v43)(__int64 *); // rdi
  __int64 v44; // [rsp+0h] [rbp-E0h]
  const char *s2; // [rsp+10h] [rbp-D0h]
  void *s2a; // [rsp+10h] [rbp-D0h]
  size_t n; // [rsp+18h] [rbp-C8h]
  size_t na; // [rsp+18h] [rbp-C8h]
  __int64 v49; // [rsp+28h] [rbp-B8h]
  const char *srca; // [rsp+38h] [rbp-A8h]
  _DWORD *src; // [rsp+38h] [rbp-A8h]
  unsigned __int8 v52; // [rsp+4Fh] [rbp-91h] BYREF
  void *v53[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v54; // [rsp+60h] [rbp-80h] BYREF
  const char *v55; // [rsp+70h] [rbp-70h] BYREF
  size_t v56; // [rsp+78h] [rbp-68h]
  __int64 v57; // [rsp+80h] [rbp-60h]
  __int64 v58; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall **v59)(__int64 *); // [rsp+90h] [rbp-50h] BYREF
  size_t v60; // [rsp+98h] [rbp-48h]
  __int64 (__fastcall *v61)(__int64 *); // [rsp+A0h] [rbp-40h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-38h]

  v5 = a2 + 32;
  v6 = a2 + 32 * a3;
  v8 = *(_QWORD *)a2;
  if ( v6 != a2 )
  {
    while ( v6 != v5 )
    {
      v5 += 32;
      if ( *(_QWORD *)(v5 - 32) != v8 )
        return 0;
    }
  }
  if ( *(_BYTE *)(a1 + 80) )
    *(_BYTE *)(a2 + 25) = 1;
  v52 = 0;
  v55 = (const char *)v8;
  v11 = a4 + 64;
  v59 = (__int64 (__fastcall **)(__int64 *))a1;
  v60 = (size_t)&v55;
  v61 = (__int64 (__fastcall *)(__int64 *))&v52;
  sub_18B7020((__int64 *)&v59, a4);
  for ( i = *(_QWORD *)(v11 + 16); v11 != i; i = sub_220EEE0(i) )
    sub_18B7020((__int64 *)&v59, i + 56);
  v9 = v52;
  if ( !v52 )
    return 0;
  if ( (*(_BYTE *)(v8 + 32) & 0xFu) - 7 > 1 )
    goto LABEL_16;
  v55 = sub_1649960(v8);
  v56 = v13;
  v59 = (__int64 (__fastcall **)(__int64 *))&v55;
  v60 = (size_t)"$merged";
  LOWORD(v61) = 773;
  sub_16E2FC0((__int64 *)v53, (__int64)&v59);
  v14 = *(_QWORD *)(v8 + 48);
  v49 = v14;
  if ( !v14 )
    goto LABEL_14;
  s2 = sub_1649960(v8);
  n = v15;
  v16 = (const void *)sub_1580C70((_QWORD *)v14);
  if ( n != v17 || n && memcmp(v16, s2, n) )
    goto LABEL_14;
  v44 = a5;
  src = (_DWORD *)sub_1633B90(*(_QWORD *)a1, v53[0], (size_t)v53[1]);
  s2a = (void *)v8;
  src[2] = *(_DWORD *)(v14 + 8);
  v31 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  na = *(_QWORD *)a1 + 8LL;
  v32 = *(_QWORD *)a1 + 24LL;
  v55 = *(const char **)(*(_QWORD *)a1 + 16LL);
  v56 = na;
  v57 = v31;
  v58 = v32;
  while ( v32 != v31 || v32 != v58 || (const char *)na != v55 || na != v56 )
  {
    v33 = (__int64 *)&v59;
    v62 = 0;
    v34 = &v59;
    v35 = &v55;
    v61 = sub_18564C0;
    v36 = sub_18564A0;
    if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
      goto LABEL_40;
    while ( 1 )
    {
      v36 = *(__int64 (__fastcall **)(__int64))((char *)v36 + (_QWORD)*v35 - 1);
LABEL_40:
      v37 = v36((__int64)v35);
      if ( v37 )
        break;
      while ( 1 )
      {
        v38 = v34[3];
        v36 = (__int64 (__fastcall *)(__int64))v34[2];
        v33 += 2;
        v34 = (__int64 (__fastcall ***)(__int64 *))v33;
        v35 = (const char **)((char *)&v55 + (_QWORD)v38);
        if ( ((unsigned __int8)v36 & 1) != 0 )
          break;
        v37 = v36((__int64)v35);
        if ( v37 )
          goto LABEL_43;
      }
    }
LABEL_43:
    if ( v49 == *(_QWORD *)(v37 + 48) )
      *(_QWORD *)(v37 + 48) = src;
    v39 = (__int64 *)&v59;
    v62 = 0;
    v40 = &v59;
    v41 = &v55;
    v61 = sub_1856470;
    v42 = sub_1856440;
    if ( ((unsigned __int8)sub_1856440 & 1) != 0 )
      goto LABEL_46;
    while ( 2 )
    {
      if ( !(unsigned __int8)v42((__int64)v41) )
      {
        while ( 1 )
        {
          v43 = v40[3];
          v42 = (__int64 (__fastcall *)(__int64))v40[2];
          v39 += 2;
          v40 = (__int64 (__fastcall ***)(__int64 *))v39;
          v41 = (const char **)((char *)&v55 + (_QWORD)v43);
          if ( ((unsigned __int8)v42 & 1) != 0 )
            break;
          if ( (unsigned __int8)v42((__int64)v41) )
            goto LABEL_50;
        }
LABEL_46:
        v42 = *(__int64 (__fastcall **)(__int64))((char *)v42 + (_QWORD)*v41 - 1);
        continue;
      }
      break;
    }
LABEL_50:
    v31 = v57;
  }
  v8 = (__int64)s2a;
  v9 = (unsigned __int8)v9;
  a5 = v44;
LABEL_14:
  *(_WORD *)(v8 + 32) = *(_WORD *)(v8 + 32) & 0xBFC0 | 0x4010;
  LOWORD(v61) = 260;
  v59 = (__int64 (__fastcall **)(__int64 *))v53;
  sub_164B780(v8, (__int64 *)&v59);
  if ( v53[0] != &v54 )
    j_j___libc_free_0(v53[0], v54 + 1);
LABEL_16:
  *(_DWORD *)a5 = 1;
  v19 = sub_1649960(v8);
  v20 = v18;
  if ( !v19 )
  {
    LOBYTE(v61) = 0;
    v23 = *(_BYTE **)(a5 + 8);
    v28 = 0;
    v59 = &v61;
LABEL_27:
    *(_QWORD *)(a5 + 16) = v28;
    v23[v28] = 0;
    v24 = v59;
    goto LABEL_24;
  }
  v55 = (const char *)v18;
  v21 = v18;
  v59 = &v61;
  if ( v18 > 0xF )
  {
    srca = v19;
    v29 = sub_22409D0(&v59, &v55, 0);
    v19 = srca;
    v59 = (__int64 (__fastcall **)(__int64 *))v29;
    v30 = (__int64 (__fastcall **)(__int64 *))v29;
    v61 = (__int64 (__fastcall *)(__int64 *))v55;
  }
  else
  {
    if ( v18 == 1 )
    {
      LOBYTE(v61) = *v19;
      v22 = &v61;
      goto LABEL_20;
    }
    if ( !v18 )
    {
      v22 = &v61;
      goto LABEL_20;
    }
    v30 = &v61;
  }
  memcpy(v30, v19, v20);
  v21 = (size_t)v55;
  v22 = v59;
LABEL_20:
  v60 = v21;
  *((_BYTE *)v22 + v21) = 0;
  v23 = *(_BYTE **)(a5 + 8);
  v24 = (__int64 (__fastcall **)(__int64 *))v23;
  if ( v59 == &v61 )
  {
    v28 = v60;
    if ( v60 )
    {
      if ( v60 == 1 )
        *v23 = (_BYTE)v61;
      else
        memcpy(v23, &v61, v60);
      v28 = v60;
      v23 = *(_BYTE **)(a5 + 8);
    }
    goto LABEL_27;
  }
  v25 = v61;
  v26 = v60;
  if ( v24 == (__int64 (__fastcall **)(__int64 *))(a5 + 24) )
  {
    *(_QWORD *)(a5 + 8) = v59;
    *(_QWORD *)(a5 + 16) = v26;
    *(_QWORD *)(a5 + 24) = v25;
    goto LABEL_29;
  }
  v27 = *(__int64 (__fastcall **)(__int64 *))(a5 + 24);
  *(_QWORD *)(a5 + 8) = v59;
  *(_QWORD *)(a5 + 16) = v26;
  *(_QWORD *)(a5 + 24) = v25;
  if ( !v24 )
  {
LABEL_29:
    v59 = &v61;
    v24 = &v61;
    goto LABEL_24;
  }
  v59 = v24;
  v61 = v27;
LABEL_24:
  v60 = 0;
  *(_BYTE *)v24 = 0;
  if ( v59 != &v61 )
    j_j___libc_free_0(v59, (char *)v61 + 1);
  return v9;
}
