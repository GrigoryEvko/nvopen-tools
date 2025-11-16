// Function: sub_A92DE0
// Address: 0xa92de0
//
__int64 __fastcall sub_A92DE0(_DWORD *s1, const char *a2, unsigned __int8 *a3, __int64 a4)
{
  size_t v5; // r13
  unsigned __int8 *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int8 *v21; // rbx
  unsigned __int8 *v22; // r12
  _BYTE *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  unsigned __int8 *v41; // [rsp+0h] [rbp-F0h]
  __int64 v42; // [rsp+10h] [rbp-E0h]
  __int64 v43; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v44; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v46; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-A8h]
  void *s1a; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-98h]
  __int64 v50; // [rsp+60h] [rbp-90h]
  __int64 v51; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v52; // [rsp+78h] [rbp-78h]
  _BYTE *v53; // [rsp+80h] [rbp-70h]
  __int64 v54; // [rsp+90h] [rbp-60h] BYREF
  __int64 v55; // [rsp+98h] [rbp-58h]
  __int64 v56; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-48h] BYREF
  __int16 v58[32]; // [rsp+B0h] [rbp-40h] BYREF

  v5 = (size_t)a2;
  v6 = a3;
  if ( a2 == (const char *)14 )
  {
    if ( *(_QWORD *)s1 == 0x707463762E65766DLL && s1[2] == 1865298998 && *((_WORD *)s1 + 6) == 25708 )
    {
      BYTE4(v51) = 0;
      v54 = sub_BD5D20(a3);
      v33 = *((_DWORD *)v6 + 1);
      v58[0] = 261;
      v55 = v34;
      s1a = *(void **)&v6[-32 * (v33 & 0x7FFFFFF)];
      v35 = sub_B33D10(a4, 3482, 0, 0, (unsigned int)&s1a, 1, v51, (__int64)&v54);
      v36 = *(_QWORD *)(a4 + 72);
      HIDWORD(v51) = 0;
      v58[0] = 257;
      v44 = v35;
      v37 = sub_BCB2A0(v36);
      BYTE4(v47) = 0;
      LODWORD(v47) = 2;
      s1a = (void *)sub_BCE1B0(v37, v47);
      v38 = sub_B33D10(a4, 3442, (unsigned int)&s1a, 1, (unsigned int)&v44, 1, v51, (__int64)&v54);
      v39 = *(_QWORD *)(a4 + 72);
      v58[0] = 257;
      v45 = v38;
      HIDWORD(v51) = 0;
      v40 = sub_BCB2A0(v39);
      BYTE4(s1a) = 0;
      LODWORD(s1a) = 4;
      v46 = sub_BCE1B0(v40, s1a);
      return sub_B33D10(a4, 3441, (unsigned int)&v46, 1, (unsigned int)&v45, 1, v51, (__int64)&v54);
    }
  }
  else if ( a2 == (const char *)40 )
  {
    if ( !(*(_QWORD *)s1 ^ 0x6C6C756D2E65766DLL | *((_QWORD *)s1 + 1) ^ 0x6572702E746E692ELL)
      && !(*((_QWORD *)s1 + 2) ^ 0x2E64657461636964LL | *((_QWORD *)s1 + 3) ^ 0x34762E3436693276LL)
      && *((_QWORD *)s1 + 4) == 0x316934762E323369LL )
    {
      goto LABEL_18;
    }
  }
  else if ( a2 == (const char *)39 )
  {
    a2 = "mve.vqdmull.predicated.v2i64.v4i32.v4i1";
    if ( !memcmp(s1, "mve.vqdmull.predicated.v2i64.v4i32.v4i1", 0x27u) )
      goto LABEL_18;
  }
  else if ( a2 == (const char *)48 )
  {
    if ( !(*(_QWORD *)s1 ^ 0x72646C762E65766DLL | *((_QWORD *)s1 + 1) ^ 0x2E7265687461672ELL)
      && !(*((_QWORD *)s1 + 2) ^ 0x6572702E65736162LL | *((_QWORD *)s1 + 3) ^ 0x2E64657461636964LL)
      && !(*((_QWORD *)s1 + 4) ^ 0x32762E3436693276LL | *((_QWORD *)s1 + 5) ^ 0x316934762E343669LL) )
    {
      goto LABEL_18;
    }
  }
  else if ( a2 == (const char *)51 )
  {
    a2 = "mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v4i1";
    if ( !memcmp(s1, "mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v4i1", 0x33u) )
      goto LABEL_18;
  }
  else if ( a2 == (const char *)56 )
  {
    if ( !(*(_QWORD *)s1 ^ 0x72646C762E65766DLL | *((_QWORD *)s1 + 1) ^ 0x2E7265687461672ELL)
      && !(*((_QWORD *)s1 + 2) ^ 0x702E74657366666FLL | *((_QWORD *)s1 + 3) ^ 0x6574616369646572LL)
      && !(*((_QWORD *)s1 + 4) ^ 0x2E34366932762E64LL | *((_QWORD *)s1 + 5) ^ 0x32762E3436693070LL)
      && *((_QWORD *)s1 + 6) == 0x316934762E343669LL )
    {
      goto LABEL_18;
    }
  }
  else if ( a2 == (const char *)53 )
  {
    a2 = "mve.vldr.gather.offset.predicated.v2i64.p0.v2i64.v4i1";
    if ( !memcmp(s1, "mve.vldr.gather.offset.predicated.v2i64.p0.v2i64.v4i1", 0x35u) )
      goto LABEL_18;
  }
  else if ( a2 == (const char *)49 )
  {
    if ( !(*(_QWORD *)s1 ^ 0x727473762E65766DLL | *((_QWORD *)s1 + 1) ^ 0x726574746163732ELL)
      && !(*((_QWORD *)s1 + 2) ^ 0x72702E657361622ELL | *((_QWORD *)s1 + 3) ^ 0x6465746163696465LL)
      && !(*((_QWORD *)s1 + 4) ^ 0x762E34366932762ELL | *((_QWORD *)s1 + 5) ^ 0x6934762E34366932LL)
      && *((_BYTE *)s1 + 48) == 49 )
    {
      goto LABEL_18;
    }
  }
  else if ( a2 == (const char *)52 )
  {
    if ( !(*(_QWORD *)s1 ^ 0x727473762E65766DLL | *((_QWORD *)s1 + 1) ^ 0x726574746163732ELL)
      && !(*((_QWORD *)s1 + 2) ^ 0x62772E657361622ELL | *((_QWORD *)s1 + 3) ^ 0x616369646572702ELL)
      && !(*((_QWORD *)s1 + 4) ^ 0x366932762E646574LL | *((_QWORD *)s1 + 5) ^ 0x2E34366932762E34LL)
      && s1[12] == 828978294 )
    {
      goto LABEL_18;
    }
  }
  else if ( a2 == (const char *)57 )
  {
    a2 = "mve.vstr.scatter.offset.predicated.p0i64.v2i64.v2i64.v4i1";
    if ( !memcmp(s1, "mve.vstr.scatter.offset.predicated.p0i64.v2i64.v2i64.v4i1", 0x39u) )
      goto LABEL_18;
  }
  else if ( a2 == (const char *)54 )
  {
    a2 = "mve.vstr.scatter.offset.predicated.p0.v2i64.v2i64.v4i1";
    if ( !memcmp(s1, "mve.vstr.scatter.offset.predicated.p0.v2i64.v2i64.v4i1", 0x36u) )
      goto LABEL_18;
  }
  a2 = (const char *)v5;
  if ( !sub_9691B0(s1, v5, "cde.vcx1q.predicated.v2i64.v4i1", 31) )
  {
    a2 = (const char *)v5;
    if ( !sub_9691B0(s1, v5, "cde.vcx1qa.predicated.v2i64.v4i1", 32) )
    {
      a2 = (const char *)v5;
      if ( !sub_9691B0(s1, v5, "cde.vcx2q.predicated.v2i64.v4i1", 31) )
      {
        a2 = (const char *)v5;
        if ( !sub_9691B0(s1, v5, "cde.vcx2qa.predicated.v2i64.v4i1", 32) )
        {
          a2 = (const char *)v5;
          if ( !sub_9691B0(s1, v5, "cde.vcx3q.predicated.v2i64.v4i1", 31) )
          {
            a2 = (const char *)v5;
            if ( !sub_9691B0(s1, v5, "cde.vcx3qa.predicated.v2i64.v4i1", 32) )
              BUG();
          }
        }
      }
    }
  }
LABEL_18:
  s1a = 0;
  v49 = 0;
  v50 = 0;
  v7 = sub_B49240(v6, a2);
  v8 = sub_BCB2A0(*(_QWORD *)(a4 + 72));
  v42 = sub_BCDA70(v8, 2);
  if ( v7 == 3516 )
  {
    v12 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
    v54 = *((_QWORD *)v6 + 1);
    v55 = *(_QWORD *)(*(_QWORD *)&v6[-32 * v12] + 8LL);
    v13 = 1;
  }
  else
  {
    if ( v7 <= 0xDBC )
    {
      if ( v7 == 3435 )
        goto LABEL_23;
      if ( v7 <= 0xD6B )
      {
        switch ( v7 )
        {
          case 0xD11u:
          case 0xD13u:
          case 0xD17u:
          case 0xD19u:
          case 0xD1Du:
          case 0xD1Fu:
            v54 = *(_QWORD *)(*(_QWORD *)&v6[32 * (1LL - (*((_DWORD *)v6 + 1) & 0x7FFFFFF))] + 8LL);
            v55 = v42;
            sub_A7BA40((__int64)&s1a, (char *)&v54, &v56);
            goto LABEL_25;
          default:
            goto LABEL_106;
        }
      }
      if ( v7 == 3512 )
        goto LABEL_23;
      if ( v7 != 3514 )
        goto LABEL_106;
      goto LABEL_35;
    }
    if ( v7 == 3588 )
      goto LABEL_35;
    if ( v7 <= 0xE04 )
    {
      if ( v7 == 3540 )
      {
LABEL_23:
        v54 = *((_QWORD *)v6 + 1);
        v9 = *(_QWORD *)(*(_QWORD *)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] + 8LL);
LABEL_24:
        v55 = v9;
        v56 = v42;
        sub_A7BA40((__int64)&s1a, (char *)&v54, &v57);
        goto LABEL_25;
      }
      if ( v7 != 3586 )
        goto LABEL_106;
LABEL_35:
      v9 = *(_QWORD *)(*(_QWORD *)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] + 8LL);
      v54 = v9;
      goto LABEL_24;
    }
    if ( v7 != 3590 )
      goto LABEL_106;
    v12 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
    v54 = *(_QWORD *)(*(_QWORD *)&v6[-32 * v12] + 8LL);
    v55 = *(_QWORD *)(*(_QWORD *)&v6[32 * (1 - v12)] + 8LL);
    v13 = 2;
  }
  v56 = *(_QWORD *)(*(_QWORD *)&v6[32 * (v13 - v12)] + 8LL);
  v57 = v42;
  sub_A7BA40((__int64)&s1a, (char *)&v54, v58);
LABEL_25:
  v10 = *v6;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  if ( v10 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0(v6);
  }
  else
  {
    v11 = -32;
    if ( v10 != 85 )
    {
      v11 = -96;
      if ( v10 != 34 )
LABEL_106:
        BUG();
    }
  }
  if ( (v6[7] & 0x80u) == 0 )
    goto LABEL_45;
  v14 = sub_BD2BC0(v6);
  v16 = v14 + v15;
  v17 = v14 + v15;
  if ( (v6[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v17 >> 4) )
      goto LABEL_105;
  }
  else if ( (unsigned int)((v16 - sub_BD2BC0(v6)) >> 4) )
  {
    if ( (v6[7] & 0x80u) != 0 )
    {
      v18 = *(_DWORD *)(sub_BD2BC0(v6) + 8);
      if ( (v6[7] & 0x80u) == 0 )
        BUG();
      v19 = sub_BD2BC0(v6);
      v11 -= 32LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
      goto LABEL_45;
    }
LABEL_105:
    BUG();
  }
LABEL_45:
  v21 = &v6[v11];
  if ( &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] == v21 )
    goto LABEL_55;
  v41 = v6;
  v22 = &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
  do
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(*(_QWORD *)v22 + 8LL);
      v43 = *(_QWORD *)v22;
      if ( (unsigned int)sub_BCB060(v24) == 1 )
        break;
      v23 = v52;
      if ( v52 != v53 )
        goto LABEL_48;
LABEL_53:
      v22 += 32;
      sub_9281F0((__int64)&v51, v23, &v43);
      if ( v21 == v22 )
        goto LABEL_54;
    }
    v25 = *(_QWORD *)(a4 + 72);
    HIDWORD(v47) = 0;
    v58[0] = 257;
    v26 = sub_BCB2A0(v25);
    BYTE4(v46) = 0;
    LODWORD(v46) = 4;
    v45 = sub_BCE1B0(v26, v46);
    v27 = sub_B33D10(a4, 3442, (unsigned int)&v45, 1, (unsigned int)&v43, 1, v47, (__int64)&v54);
    v58[0] = 257;
    v44 = v27;
    HIDWORD(v47) = 0;
    v45 = v42;
    v28 = sub_B33D10(a4, 3441, (unsigned int)&v45, 1, (unsigned int)&v44, 1, (unsigned int)v47, (__int64)&v54);
    v23 = v52;
    v43 = v28;
    if ( v52 == v53 )
      goto LABEL_53;
LABEL_48:
    if ( v23 )
    {
      *(_QWORD *)v23 = v43;
      v23 = v52;
    }
    v22 += 32;
    v52 = v23 + 8;
  }
  while ( v21 != v22 );
LABEL_54:
  v6 = v41;
LABEL_55:
  v29 = sub_BD5D20(v6);
  v58[0] = 261;
  v55 = v30;
  BYTE4(v47) = 0;
  v54 = v29;
  v31 = sub_B33D10(a4, v7, (_DWORD)s1a, (v49 - (__int64)s1a) >> 3, v51, (__int64)&v52[-v51] >> 3, v47, (__int64)&v54);
  if ( v51 )
    j_j___libc_free_0(v51, &v53[-v51]);
  if ( s1a )
    j_j___libc_free_0(s1a, v50 - (_QWORD)s1a);
  return v31;
}
