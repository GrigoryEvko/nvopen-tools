// Function: sub_87AFA0
// Address: 0x87afa0
//
__int64 __fastcall sub_87AFA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 i; // r15
  __int64 v4; // rax
  __m128i *v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 j; // rax
  __int64 v11; // r11
  unsigned __int8 v12; // al
  char v13; // r14
  _DWORD *v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // rdi
  const __m128i *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rax
  char k; // dl
  __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rdx
  __int64 v28; // rdi
  __int64 v29; // rax
  char m; // dl
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  char n; // dl
  __int64 v38; // rdi
  const char *v39; // rsi
  const __m128i *v40; // r12
  __int64 v41; // rdi
  const char *v42; // r13
  const char *v43; // rsi
  __int64 v44; // r13
  __int64 v45; // r12
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // r10
  char v49; // al
  _BOOL4 v50; // eax
  __int64 v51; // r10
  __int64 v52; // rsi
  __int64 v53; // r8
  char v54; // al
  __int64 v55; // rdi
  char v56; // al
  __int64 jj; // r13
  _QWORD *v58; // rdi
  int v59; // r14d
  _DWORD *v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // rax
  unsigned __int8 *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rsi
  char v66; // al
  __int64 v67; // r14
  __int64 v68; // r13
  __int64 ii; // r12
  __int64 v70; // rdx
  _QWORD **v71; // rax
  _QWORD *v72; // rdi
  __int64 v73; // rax
  _DWORD *v74; // rax
  int v75; // edx
  __int64 v76; // rdi
  __int64 v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // [rsp+8h] [rbp-4B8h]
  __int64 v80; // [rsp+20h] [rbp-4A0h]
  __int64 v81; // [rsp+20h] [rbp-4A0h]
  _QWORD *v82; // [rsp+28h] [rbp-498h]
  __int64 v83; // [rsp+28h] [rbp-498h]
  _DWORD *v84; // [rsp+28h] [rbp-498h]
  __int64 v85; // [rsp+30h] [rbp-490h]
  FILE *v86; // [rsp+30h] [rbp-490h]
  __int64 v88; // [rsp+48h] [rbp-478h]
  __int64 v89; // [rsp+50h] [rbp-470h]
  __int64 *v91; // [rsp+60h] [rbp-460h]
  _DWORD *v92; // [rsp+60h] [rbp-460h]
  __int64 v93; // [rsp+68h] [rbp-458h]
  __int64 v94; // [rsp+78h] [rbp-448h] BYREF
  _QWORD *v95; // [rsp+80h] [rbp-440h] BYREF
  __int64 v96; // [rsp+88h] [rbp-438h] BYREF
  _DWORD v97[40]; // [rsp+90h] [rbp-430h] BYREF
  _QWORD *v98[20]; // [rsp+130h] [rbp-390h] BYREF
  __m128i v99; // [rsp+1D0h] [rbp-2F0h] BYREF
  char v100; // [rsp+1E0h] [rbp-2E0h]
  __m128i v101; // [rsp+330h] [rbp-190h] BYREF
  char v102; // [rsp+340h] [rbp-180h]
  __int64 v103[18]; // [rsp+3C0h] [rbp-100h] BYREF
  __int64 v104; // [rsp+450h] [rbp-70h]
  char v105; // [rsp+46Dh] [rbp-53h]
  char v106; // [rsp+470h] [rbp-50h]

  v2 = a1[2];
  v93 = qword_4D03C50;
  v89 = qword_4F06BC0;
  for ( i = qword_4F06BC0;
        *(_BYTE *)i != 1 || *(_BYTE *)(i + 8) != 23 || *(_BYTE *)(*(_QWORD *)(i + 16) + 28LL) != 17;
        i = *(_QWORD *)(i + 32) )
  {
    ;
  }
  v4 = *(_QWORD *)(i + 24);
  *(_QWORD *)(i + 24) = 0;
  v98[0] = 0;
  v88 = v4;
  v99.m128i_i64[0] = 0;
  qword_4F06BC0 = i;
  v91 = (__int64 *)(v2 + 64);
  v5 = &v101;
  sub_6E1E00(4u, (__int64)&v101, 0, 0);
  v9 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
  for ( j = *(_QWORD *)(v2 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 8LL);
  if ( !v11 )
    goto LABEL_17;
  v12 = *(_BYTE *)(v9 + 18);
  v85 = v11;
  *(_BYTE *)(v9 + 18) = v12 | 0x80;
  v13 = v12 >> 7 << 7;
  sub_877320(a2, (__int64 *)v98);
  v5 = (__m128i *)(v2 + 64);
  sub_6C5750(v85, (__int64)v91, 0, 0, 0, 0, 0, 1, 0, 1u, (__int64)v98[0], 0, 0, 0, 0, 0, 0, 0, v99.m128i_i64, 0, 0);
  v9 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 18LL) = *(_BYTE *)(qword_4D03C50 + 18LL) & 0x7F | v13;
  if ( (*(_BYTE *)(v9 + 19) & 1) != 0 )
  {
    v99.m128i_i64[0] = 0;
LABEL_11:
    v5 = (__m128i *)(v2 + 64);
    v14 = v97;
    v15 = sub_6EB0E0(*(_QWORD *)(v2 + 120), (int)v91, (__int64)v97);
    if ( v97[0] || !v15 )
    {
      v99.m128i_i64[0] = sub_6EAFA0(0);
      v16 = v99.m128i_i64[0];
    }
    else
    {
      v5 = 0;
      v99.m128i_i64[0] = sub_6F5430(v15, 0, 0, 0, 1, 0, 0, 0, 1u, 0, (__int64)v91);
      v16 = v99.m128i_i64[0];
    }
    goto LABEL_19;
  }
  v16 = v99.m128i_i64[0];
  if ( !v99.m128i_i64[0] )
    goto LABEL_11;
  v14 = v97;
  if ( *(_BYTE *)(v99.m128i_i64[0] + 48) == 5 )
  {
    sub_732AE0(*(_QWORD *)(v99.m128i_i64[0] + 56), (__int64)v91, v9, v6, v7, v8);
LABEL_17:
    v16 = v99.m128i_i64[0];
    if ( !v99.m128i_i64[0] )
      goto LABEL_11;
    v14 = v97;
  }
LABEL_19:
  sub_6E2920(v16, v5, v9, v6, v7, v8);
  v17 = *(_QWORD *)(v2 + 120);
  sub_6EB360(v99.m128i_i64[0], v17, v17, v91);
  v18 = v99.m128i_i64[0];
  *(_QWORD *)(v99.m128i_i64[0] + 8) = v2;
  *(_QWORD *)(v2 + 184) = v18;
  *(_BYTE *)(v2 + 177) = 2;
  v19 = v98[0];
  sub_6E1990(v98[0]);
  sub_6E2B30((__int64)v19, v17);
  qword_4D03C50 = v93;
  sub_6E1E00(4u, (__int64)v97, 0, 0);
  v20 = (const __m128i *)"initial_suspend";
  v21 = (__int64)&v99;
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
  sub_8773E0(&v99, "initial_suspend", v2, 1, 1);
  if ( !v100 )
    goto LABEL_23;
  v22 = v99.m128i_i64[0];
  for ( k = *(_BYTE *)(v99.m128i_i64[0] + 140); k == 12; k = *(_BYTE *)(v22 + 140) )
    v22 = *(_QWORD *)(v22 + 160);
  if ( !k )
    goto LABEL_23;
  v28 = sub_6FE2B0(&v99);
  a1[7] = v28;
  if ( v28 )
    sub_7304E0(v28);
  sub_6E2B30(v28, (__int64)"initial_suspend");
  sub_6E1E00(4u, (__int64)v97, 0, 0);
  v20 = (const __m128i *)"final_suspend";
  v21 = (__int64)&v99;
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
  sub_8773E0(&v99, "final_suspend", v2, 1, 0);
  if ( !v100 )
    goto LABEL_23;
  v29 = v99.m128i_i64[0];
  for ( m = *(_BYTE *)(v99.m128i_i64[0] + 140); m == 12; m = *(_BYTE *)(v29 + 140) )
    v29 = *(_QWORD *)(v29 + 160);
  if ( !m )
    goto LABEL_23;
  v31 = sub_6FE2B0(&v99);
  a1[8] = v31;
  if ( v31 )
  {
    sub_7304E0(v31);
    v31 = a1[8];
    if ( v31 )
    {
      if ( *(_BYTE *)(v31 + 24) && (unsigned int)sub_731B40(v31, "final_suspend", v32, v33, v34, v35) )
      {
        v63 = sub_694FD0(*(_QWORD *)(v2 + 120), "final_suspend", &v101);
        v31 = 2981;
        v20 = (const __m128i *)(v63 + 48);
        sub_6854C0(0xBA5u, (FILE *)(v63 + 48), (__int64)v63);
      }
    }
  }
  sub_6E2B30(v31, (__int64)v20);
  if ( dword_4D048B8 )
  {
    sub_6E1E00(4u, (__int64)v97, 0, 0);
    v20 = (const __m128i *)"unhandled_exception";
    v21 = (__int64)&v99;
    *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
    sub_8773E0(&v99, "unhandled_exception", v2, 0, 0);
    if ( v100 )
    {
      v36 = v99.m128i_i64[0];
      for ( n = *(_BYTE *)(v99.m128i_i64[0] + 140); n == 12; n = *(_BYTE *)(v36 + 140) )
        v36 = *(_QWORD *)(v36 + 160);
      if ( n )
      {
        v38 = sub_6FE2B0(&v99);
        a1[9] = v38;
        if ( v38 )
          sub_7304E0(v38);
        sub_6E2B30(v38, (__int64)"unhandled_exception");
        goto LABEL_46;
      }
    }
LABEL_23:
    v92 = (_DWORD *)qword_4D03C50;
    goto LABEL_24;
  }
LABEL_46:
  sub_6E1E00(4u, (__int64)v97, 0, 0);
  v39 = "get_return_object";
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 8u;
  sub_8773E0(&v99, "get_return_object", v2, 0, 0);
  v40 = *(const __m128i **)(*(_QWORD *)(a2 + 152) + 160LL);
  if ( (unsigned int)sub_8D3A70(v40) )
  {
    v39 = (const char *)v40;
    sub_8470D0((__int64)&v99, v40, 0, 2u, 0x78u, 0, &v94);
  }
  else if ( !(unsigned int)sub_8D2600(v40) )
  {
    v39 = (const char *)v40;
    sub_843C40(&v99, (__int64)v40, 0, 0, 1, 2u, 0x78u);
  }
  v41 = sub_6FE2B0(&v99);
  a1[10] = v41;
  if ( v41 )
    sub_7304E0(v41);
  sub_6E2B30(v41, (__int64)v39);
  v42 = *(const char **)(a1[2] + 120LL);
  v86 = (FILE *)(a1 + 14);
  v79 = (__int64)v42;
  v92 = (_DWORD *)qword_4D03C50;
  sub_6E1E00(4u, (__int64)v98, 0, 0);
  v80 = sub_7D3790(1u, v42);
  v43 = v42;
  v44 = sub_7D3790(2u, v42);
  v45 = sub_879C70("get_return_object_on_allocation_failure", v43, 0x10u);
  v46 = sub_73A8E0(0, byte_4F06A51[0]);
  sub_6E70E0(v46, (__int64)&v101);
  v47 = sub_6E3060(&v101);
  v48 = v80;
  v82 = (_QWORD *)v47;
  if ( !v80 )
  {
    v65 = sub_7D3810(1u);
    v66 = *(_BYTE *)(v65 + 80);
    if ( v66 == 16 )
    {
      v65 = **(_QWORD **)(v65 + 88);
      v66 = *(_BYTE *)(v65 + 80);
    }
    if ( v66 == 24 )
    {
      v65 = *(_QWORD *)(v65 + 88);
      v66 = *(_BYTE *)(v65 + 80);
    }
    if ( v66 != 17 )
    {
      v67 = v44;
      v68 = v45;
      ii = v65;
      goto LABEL_108;
    }
    v77 = *(_QWORD *)(v65 + 88);
    if ( v77 )
    {
      v66 = *(_BYTE *)(v77 + 80);
      v67 = v44;
      v68 = v45;
      for ( ii = v77; ; v66 = *(_BYTE *)(ii + 80) )
      {
LABEL_108:
        v70 = *(_QWORD *)(ii + 88);
        if ( v66 == 20 )
          v70 = *(_QWORD *)(v70 + 176);
        v71 = **(_QWORD ****)(*(_QWORD *)(v70 + 152) + 168LL);
        if ( v71 )
        {
          v72 = *v71;
          if ( v68 )
          {
            if ( v72 && !*v72 && sub_646150((__int64)v72) )
            {
LABEL_113:
              v52 = ii;
              v45 = v68;
              v44 = v67;
              v14 = v97;
              goto LABEL_114;
            }
          }
          else if ( !v72 )
          {
            goto LABEL_113;
          }
        }
        ii = *(_QWORD *)(ii + 8);
        if ( !ii )
          break;
      }
      v45 = v68;
      v44 = v67;
      v14 = v97;
    }
    sub_6851C0(0xBA0u, v86);
    goto LABEL_59;
  }
  v49 = *(_BYTE *)(v80 + 80);
  if ( v49 == 16 )
  {
    v48 = **(_QWORD **)(v80 + 88);
    v49 = *(_BYTE *)(v48 + 80);
  }
  if ( v49 == 24 )
    v48 = *(_QWORD *)(v48 + 88);
  v81 = v48;
  sub_877320(a2, (__int64 *)&v95);
  *v82 = v95;
  v50 = sub_84AA50(v81, 3, 0, 0, (__int64)v82, 0, 0);
  v51 = v81;
  if ( !v50 )
  {
    sub_6E1990(v95);
    v51 = v81;
    *v82 = 0;
  }
  v52 = sub_84AC10(v51, 0, 0, 0, 0, v82, 0, 0, 0, 0, 0, 3, v86, 0, 0, 0, 0, 0, 0, &v96);
  if ( !v52 )
  {
LABEL_59:
    sub_6E1990(v82);
    goto LABEL_60;
  }
LABEL_114:
  sub_8767A0(4, v52, v86, 0);
  sub_6E1990(v82);
  v73 = *(_QWORD *)(v52 + 88);
  if ( *(_BYTE *)(v52 + 80) == 20 )
    v73 = *(_QWORD *)(v73 + 176);
  a1[12] = v73;
LABEL_60:
  if ( !v44 )
    v44 = sub_7D3810(2u);
  v54 = *(_BYTE *)(v44 + 80);
  if ( v54 == 16 )
  {
    v44 = **(_QWORD **)(v44 + 88);
    v54 = *(_BYTE *)(v44 + 80);
  }
  if ( v54 == 24 )
  {
    v44 = *(_QWORD *)(v44 + 88);
    v54 = *(_BYTE *)(v44 + 80);
  }
  if ( v54 == 17 )
  {
    v44 = *(_QWORD *)(v44 + 88);
    if ( !v44 )
      goto LABEL_73;
    v83 = 0;
    do
    {
      v76 = *(_QWORD *)(v44 + 88);
      if ( *(_BYTE *)(v44 + 80) == 20 )
        v76 = *(_QWORD *)(v76 + 176);
      if ( sub_87ADD0(v76, &v95, &v96, &v101, v53) )
      {
        v53 = (unsigned int)v96;
        if ( !(_DWORD)v96 )
        {
          if ( (_DWORD)v95 )
            goto LABEL_95;
          v83 = v44;
        }
      }
      v44 = *(_QWORD *)(v44 + 8);
    }
    while ( v44 );
    if ( !v83 )
      goto LABEL_73;
    v44 = v83;
  }
  else
  {
    if ( (unsigned __int8)(v54 - 10) > 1u && v54 != 20 )
      goto LABEL_73;
    v55 = *(_QWORD *)(v44 + 88);
    if ( v54 == 20 )
      v55 = *(_QWORD *)(v55 + 176);
    if ( !sub_87ADD0(v55, &v95, &v96, &v101, v53) || (_DWORD)v96 )
    {
LABEL_73:
      v20 = (const __m128i *)(a1 + 14);
      v21 = 2977;
      sub_6851C0(0xBA1u, v86);
      goto LABEL_74;
    }
  }
LABEL_95:
  v20 = (const __m128i *)v44;
  v21 = 4;
  sub_8767A0(4, v44, v86, 0);
  if ( *(_BYTE *)(v44 + 80) == 20 )
    v64 = *(_QWORD *)(*(_QWORD *)(v44 + 88) + 176LL);
  else
    v64 = *(_QWORD *)(v44 + 88);
  a1[13] = v64;
LABEL_74:
  if ( v45 )
  {
    v56 = *(_BYTE *)(v45 + 80);
    if ( v56 == 17 )
    {
      v45 = *(_QWORD *)(v45 + 88);
      if ( !v45 )
      {
LABEL_93:
        v20 = (const __m128i *)v79;
        v21 = 3194;
        sub_685380(0xC7Au, v79);
        goto LABEL_94;
      }
      v56 = *(_BYTE *)(v45 + 80);
    }
    jj = 0;
    v58 = 0;
    v59 = 0;
    v60 = v97;
    while ( 1 )
    {
      if ( v56 == 10 )
      {
        for ( jj = *(_QWORD *)(*(_QWORD *)(v45 + 88) + 152LL); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
          ;
        v61 = *(__int64 **)(jj + 168);
        if ( (*((_BYTE *)v61 + 21) & 1) == 0 )
        {
          v62 = *v61;
          if ( !v62 || (*(_BYTE *)(v62 + 32) & 4) != 0 )
          {
            if ( v58 )
            {
              v59 = 1;
            }
            else
            {
              v84 = v60;
              v78 = sub_7312D0(*(_QWORD *)(v45 + 88));
              v60 = v84;
              v58 = v78;
            }
          }
        }
      }
      v45 = *(_QWORD *)(v45 + 8);
      if ( !v45 )
        break;
      v56 = *(_BYTE *)(v45 + 80);
    }
    v74 = v60;
    v75 = v59;
    v14 = v74;
    if ( v58 && !v75 )
    {
      v20 = (const __m128i *)jj;
      sub_701D00(v58, jj, 0, 0, 0, 1, 0, 0, 1, 0, 0, (__int64 *)&dword_4F077C8, v86, &dword_4F077C8, &v101, 0, 0);
      if ( (unsigned int)sub_8D3A70(*(_QWORD *)(*(_QWORD *)(a2 + 152) + 160LL)) )
      {
        v20 = *(const __m128i **)(*(_QWORD *)(a2 + 152) + 160LL);
        sub_8470D0((__int64)&v101, v20, 0, 2u, 0x78u, 0, &v96);
      }
      v21 = v103[0];
      if ( v102 != 1 )
      {
        if ( v102 != 2 )
        {
LABEL_123:
          a1[11] = 0;
          goto LABEL_94;
        }
        v21 = v104;
        if ( v104 )
        {
          a1[11] = v104;
LABEL_144:
          sub_7304E0(v21);
          goto LABEL_94;
        }
        if ( v105 != 12 || v106 != 1 )
          goto LABEL_123;
        v21 = (__int64)sub_72E9A0((__int64)v103);
      }
      a1[11] = v21;
      if ( !v21 )
        goto LABEL_94;
      goto LABEL_144;
    }
    goto LABEL_93;
  }
LABEL_94:
  sub_6E2B30(v21, (__int64)v20);
  qword_4D03C50 = v92;
LABEL_24:
  if ( v92 == v14 )
  {
    sub_6E2B30(v21, (__int64)v20);
    v24 = *(_QWORD *)(i + 24);
    if ( v24 )
      goto LABEL_26;
LABEL_89:
    v26 = (_QWORD *)(i + 24);
    goto LABEL_28;
  }
  v24 = *(_QWORD *)(i + 24);
  if ( !v24 )
    goto LABEL_89;
  do
  {
LABEL_26:
    v25 = v24;
    v24 = *(_QWORD *)(v24 + 32);
  }
  while ( v24 );
  v26 = (_QWORD *)(v25 + 32);
LABEL_28:
  *v26 = v88;
  qword_4F06BC0 = v89;
  qword_4D03C50 = v93;
  return v93;
}
