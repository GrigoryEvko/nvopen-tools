// Function: sub_E1E1C0
// Address: 0xe1e1c0
//
__int64 __fastcall sub_E1E1C0(char **a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  signed __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 result; // rax
  char *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  char *v13; // r12
  char *v14; // r13
  char *v15; // rdx
  char *v16; // rcx
  _OWORD *v17; // rcx
  __int64 v18; // rax
  char *v19; // rax
  char *v20; // rsi
  __int64 v21; // r13
  unsigned __int64 v22; // rsi
  char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  void *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rbx
  char *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  char v41; // al
  void *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r13
  char *v45; // rax
  char v46; // dl
  __int64 v47; // r8
  __int64 v48; // rsi
  _OWORD *v49; // rdi
  char *v50; // rax
  __int64 v51; // rdx
  char v52; // dl
  char v53; // dl
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  signed __int64 v57; // r9
  char *v58; // rax
  char v59; // si
  __int64 v60; // rbx
  __int64 v61; // rax
  char *v62; // rdi
  char v63; // r14
  __int64 v64; // rax
  __int64 v65; // [rsp+0h] [rbp-E0h]
  signed __int64 v66; // [rsp+8h] [rbp-D8h]
  char v67; // [rsp+10h] [rbp-D0h]
  __int64 v68; // [rsp+10h] [rbp-D0h]
  __int64 v69; // [rsp+18h] [rbp-C8h]
  __int64 v70; // [rsp+20h] [rbp-C0h]
  char *v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+38h] [rbp-A8h] BYREF
  char **v73; // [rsp+40h] [rbp-A0h]
  __int64 v74; // [rsp+48h] [rbp-98h]
  _QWORD v75[3]; // [rsp+50h] [rbp-90h] BYREF
  _OWORD v76[4]; // [rsp+68h] [rbp-78h] BYREF
  _BYTE v77[56]; // [rsp+A8h] [rbp-38h] BYREF

  if ( a2 )
    a1[84] = a1[83];
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Ut") )
  {
    v6 = sub_E0DEF0(a1, 0);
    v8 = v7;
    result = 0;
    v10 = *a1;
    if ( a1[1] != *a1 && *v10 == 95 )
    {
      v51 = (__int64)(v10 + 1);
      *a1 = (char *)v51;
      result = sub_E0E790((__int64)(a1 + 102), 32, v51, v3, v4, v5);
      if ( result )
      {
        v52 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 16) = v6;
        *(_WORD *)(result + 8) = 16435;
        *(_QWORD *)(result + 24) = v8;
        *(_BYTE *)(result + 10) = v52 & 0xF0 | 5;
        *(_QWORD *)result = &unk_49E00E8;
      }
    }
    return result;
  }
  if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Ul") )
  {
    if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Ub") )
      return 0;
    sub_E0DEF0(a1, 0);
    v50 = *a1;
    if ( *a1 == a1[1] || *v50 != 95 )
      return 0;
    *a1 = v50 + 1;
    return sub_E0FD70((__int64)(a1 + 102), "'block-literal'");
  }
  v13 = a1[84];
  v14 = a1[83];
  v15 = v77;
  v16 = a1[98];
  v73 = a1;
  v75[2] = v77;
  v71 = v16;
  v17 = v76;
  v18 = (v13 - v14) >> 3;
  v75[0] = v76;
  a1[98] = (char *)v18;
  v74 = v18;
  v75[1] = v76;
  memset(v76, 0, sizeof(v76));
  if ( v13 != a1[85] )
    goto LABEL_9;
  v60 = 16 * v18;
  if ( v14 == (char *)(a1 + 86) )
  {
    v62 = (char *)malloc(v60, 2, v77, v76, v11, v12);
    if ( v62 )
    {
      if ( v13 != v14 )
        v62 = (char *)memmove(v62, v14, v13 - v14);
      a1[83] = v62;
      goto LABEL_49;
    }
LABEL_56:
    abort();
  }
  v61 = realloc(v14);
  a1[83] = (char *)v61;
  v62 = (char *)v61;
  if ( !v61 )
    goto LABEL_56;
LABEL_49:
  v13 = &v62[v13 - v14];
  a1[85] = &v62[v60];
LABEL_9:
  a1[84] = v13 + 8;
  *(_QWORD *)v13 = v75;
  v19 = *a1;
  v20 = a1[1];
  v21 = (a1[3] - a1[2]) >> 3;
  if ( v20 != *a1 )
  {
    v22 = v20 - v19;
    do
    {
      if ( *v19 != 84 )
        break;
      if ( v22 <= 1 )
        break;
      v23 = (char *)memchr("yptnk", v19[1], 5u);
      if ( !v23 )
        break;
      v15 = "";
      if ( v23 == "" )
        break;
      result = sub_E1DAD0((__int64)a1, (__int64)v75);
      v72 = result;
      if ( !result )
        goto LABEL_30;
      sub_E18380((__int64)(a1 + 2), &v72, v24, v25, v26, v27);
      v15 = a1[1];
      v19 = *a1;
      v22 = v15 - *a1;
    }
    while ( *a1 != v15 );
  }
  v28 = sub_E11E80(a1, v21, (__int64)v15, (__int64)v17, v11, v12);
  v30 = v29;
  if ( !v29 )
    a1[84] -= 8;
  v69 = 0;
  v31 = *a1;
  if ( *a1 != a1[1] && *v31 == 81 )
  {
    v63 = *((_BYTE *)a1 + 778);
    *((_BYTE *)a1 + 778) = 1;
    *a1 = v31 + 1;
    v64 = sub_E18BB0((__int64)a1);
    *((_BYTE *)a1 + 778) = v63;
    v69 = v64;
    if ( !v64 )
      goto LABEL_29;
  }
  v32 = 1;
  if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 1u, "v") )
  {
    while ( 1 )
    {
      result = sub_E1AEA0((__int64)a1, v32, v33, v34, v35);
      v72 = result;
      if ( !result )
        goto LABEL_30;
      v32 = (__int64)&v72;
      sub_E18380((__int64)(a1 + 2), &v72, v37, v38, v39, v40);
      if ( a1[1] != *a1 )
      {
        v41 = **a1;
        if ( v41 == 69 || v41 == 81 )
          break;
      }
    }
  }
  v42 = sub_E11E80(a1, v21, v33, v34, v35, v36);
  v44 = v43;
  v45 = *a1;
  if ( *a1 == a1[1] )
    goto LABEL_29;
  v46 = *v45;
  v47 = 0;
  if ( *v45 == 81 )
  {
    v53 = *((_BYTE *)a1 + 778);
    *((_BYTE *)a1 + 778) = 1;
    *a1 = v45 + 1;
    v67 = v53;
    v47 = sub_E18BB0((__int64)a1);
    *((_BYTE *)a1 + 778) = v67;
    if ( !v47 )
      goto LABEL_29;
    v45 = *a1;
    if ( *a1 == a1[1] )
      goto LABEL_29;
    v46 = *v45;
  }
  if ( v46 == 69 )
  {
    v68 = v47;
    *a1 = v45 + 1;
    v57 = sub_E0DEF0(a1, 0);
    v58 = *a1;
    if ( *a1 != a1[1] && *v58 == 95 )
    {
      v65 = v54;
      *a1 = v58 + 1;
      v66 = v57;
      result = sub_E0E790((__int64)(a1 + 102), 80, v54, v55, v56, v57);
      if ( result )
      {
        v59 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 16) = v28;
        *(_WORD *)(result + 8) = 16436;
        *(_QWORD *)(result + 24) = v30;
        *(_QWORD *)(result + 32) = v69;
        *(_BYTE *)(result + 10) = v59 & 0xF0 | 5;
        *(_QWORD *)(result + 40) = v42;
        *(_QWORD *)(result + 48) = v44;
        *(_QWORD *)result = &unk_49E0148;
        *(_QWORD *)(result + 56) = v68;
        *(_QWORD *)(result + 64) = v66;
        *(_QWORD *)(result + 72) = v65;
      }
      goto LABEL_30;
    }
  }
LABEL_29:
  result = 0;
LABEL_30:
  v48 = (__int64)&v73[83][8 * v74];
  v49 = (_OWORD *)v75[0];
  v73[84] = (char *)v48;
  if ( v49 != v76 )
  {
    v70 = result;
    _libc_free(v49, v48);
    result = v70;
  }
  a1[98] = v71;
  return result;
}
