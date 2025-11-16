// Function: sub_2CB94C0
// Address: 0x2cb94c0
//
__int64 __fastcall sub_2CB94C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  int v4; // r14d
  char *v6; // rax
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned int v14; // ebx
  __int64 v15; // rdi
  char v16; // r12
  __int64 v17; // rax
  size_t v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rax
  size_t v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // r15
  int v25; // edx
  char v26; // al
  __int64 v27; // rdi
  __int64 v28; // rax
  size_t v29; // rdx
  unsigned __int64 v30; // rdx
  const char *v31; // r9
  size_t v32; // r8
  unsigned __int64 v33; // rax
  _QWORD *v34; // rdx
  size_t v35; // rdx
  char *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  char *v41; // rcx
  char *v42; // rax
  char *v43; // rdi
  __int64 v44; // [rsp+0h] [rbp-B0h]
  size_t n; // [rsp+8h] [rbp-A8h]
  size_t na; // [rsp+8h] [rbp-A8h]
  void *src; // [rsp+10h] [rbp-A0h]
  char srca; // [rsp+10h] [rbp-A0h]
  const char *srcb; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+20h] [rbp-90h]
  char v52; // [rsp+20h] [rbp-90h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  __int64 v54; // [rsp+28h] [rbp-88h]
  unsigned __int64 v55; // [rsp+38h] [rbp-78h] BYREF
  _QWORD *v56; // [rsp+40h] [rbp-70h] BYREF
  __int64 v57; // [rsp+48h] [rbp-68h]
  _BYTE v58[16]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v59; // [rsp+60h] [rbp-50h] BYREF
  size_t v60; // [rsp+68h] [rbp-48h]
  _QWORD v61[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 32);
  v54 = a2 + 24;
  v50 = 0;
  if ( v2 != a2 + 24 )
  {
    while ( 1 )
    {
      v3 = v2 - 56;
      if ( !v2 )
        v3 = 0;
      if ( sub_B2FC80(v3) )
        goto LABEL_11;
      v59 = *(_QWORD *)(v3 + 120);
      if ( (unsigned __int8)sub_A73ED0(&v59, 48) )
        goto LABEL_11;
      v4 = 1;
      if ( (unsigned __int8)sub_CE9220(v3) )
        goto LABEL_14;
      if ( !(unsigned __int8)sub_CE7ED0(v3, "wroimage", 8u, &v59)
        && !(unsigned __int8)sub_CE7ED0(v3, "rdoimage", 8u, &v59)
        && !(unsigned __int8)sub_CE7ED0(v3, "sampler", 7u, &v59) )
      {
        break;
      }
      v59 = *(_QWORD *)(v3 + 120);
      if ( (unsigned __int8)sub_A73ED0(&v59, 31) )
      {
        v4 = 2;
        goto LABEL_15;
      }
LABEL_9:
      *(_WORD *)(v3 + 32) = *(_WORD *)(v3 + 32) & 0xBCC0 | 0x4007;
LABEL_10:
      sub_B2CD30(v3, 3);
      v50 = 1;
LABEL_11:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v54 == v2 )
        return v50;
    }
    v59 = *(_QWORD *)(v3 + 120);
    if ( (unsigned __int8)sub_A73ED0(&v59, 31) )
      goto LABEL_11;
    v10 = *(_QWORD *)(v3 + 40) + 312LL;
    if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v3, 31, v8, v9);
      v11 = *(_QWORD *)(v3 + 96);
      if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
      {
        v53 = *(_QWORD *)(v3 + 96);
        sub_B2C6D0(v3, 31, v37, v38);
        v12 = *(_QWORD *)(v3 + 96);
        v11 = v53;
      }
      else
      {
        v12 = *(_QWORD *)(v3 + 96);
      }
    }
    else
    {
      v11 = *(_QWORD *)(v3 + 96);
      v12 = v11;
    }
    v51 = v12 + 40LL * *(_QWORD *)(v3 + 104);
    if ( v51 == v11 )
      goto LABEL_33;
    v44 = v2;
    v13 = v11;
    v14 = 0;
    n = v3;
    do
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v13 + 8);
        if ( !(unsigned __int8)sub_B2D680(v13) )
          break;
        v15 = v13;
        v13 += 40;
        src = (void *)sub_B2BD20(v15);
        v16 = sub_AE5020(v10, (__int64)src);
        v17 = sub_9208B0(v10, (__int64)src);
        v60 = v18;
        v59 = ((1LL << v16) + ((unsigned __int64)(v17 + 7) >> 3) - 1) >> v16 << v16;
        v14 += sub_CA1930(&v59);
        if ( v13 == v51 )
          goto LABEL_32;
      }
      srca = sub_AE5020(v10, v19);
      v20 = sub_9208B0(v10, v19);
      v60 = v21;
      v59 = ((1LL << srca) + ((unsigned __int64)(v20 + 7) >> 3) - 1) >> srca << srca;
      v22 = sub_CA1930(&v59);
      if ( v22 < 4 )
        v22 = 4;
      v13 += 40;
      v14 += v22;
    }
    while ( v13 != v51 );
LABEL_32:
    v23 = v14;
    v3 = n;
    v2 = v44;
    if ( v23 <= 0x180 )
    {
LABEL_33:
      v24 = **(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL);
      v25 = *(unsigned __int8 *)(v24 + 8);
      if ( (_BYTE)v25 != 12
        && (unsigned __int8)v25 > 3u
        && (_BYTE)v25 != 5
        && (v25 & 0xFB) != 0xA
        && (v25 & 0xFD) != 4
        && ((unsigned __int8)(*(_BYTE *)(v24 + 8) - 15) > 3u && v25 != 20
         || !(unsigned __int8)sub_BCEBA0(**(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL), 0)) )
      {
        goto LABEL_11;
      }
      v26 = sub_AE5020(v10, v24);
      v27 = v10;
      v4 = 8;
      v52 = v26;
      v28 = sub_9208B0(v27, v24);
      v60 = v29;
      v59 = ((1LL << v52) + ((unsigned __int64)(v28 + 7) >> 3) - 1) >> v52 << v52;
      if ( (unsigned __int64)sub_CA1930(&v59) <= 0x90 )
        goto LABEL_11;
    }
    else
    {
      v4 = 7;
    }
LABEL_14:
    v59 = *(_QWORD *)(v3 + 120);
    if ( !(unsigned __int8)sub_A73ED0(&v59, 31) )
      goto LABEL_10;
LABEL_15:
    v6 = (char *)sub_C94E20((__int64)qword_4F86430);
    if ( v6 )
      v7 = *v6;
    else
      v7 = qword_4F86430[2];
    if ( v7 )
    {
LABEL_18:
      sub_B2D470(v3, 31);
      if ( v4 != 2 )
        goto LABEL_10;
      goto LABEL_9;
    }
    v58[0] = 0;
    v56 = v58;
    v57 = 0;
    sub_2C75640((__int64 *)&v59, v3);
    sub_2241490((unsigned __int64 *)&v56, (char *)v59, v60);
    if ( (_QWORD *)v59 != v61 )
      j_j___libc_free_0(v59);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v57) <= 0xA
      || (sub_2241490((unsigned __int64 *)&v56, ": Warning: ", 0xBu), (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v57) <= 8) )
    {
LABEL_72:
      sub_4262D8((__int64)"basic_string::append");
    }
    sub_2241490((unsigned __int64 *)&v56, "Function ", 9u);
    v31 = sub_BD5D20(v3);
    v32 = v30;
    if ( !v31 )
    {
      v60 = 0;
      v35 = 0;
      LOBYTE(v61[0]) = 0;
      v59 = (unsigned __int64)v61;
      v36 = (char *)v61;
      goto LABEL_58;
    }
    v55 = v30;
    v33 = v30;
    v59 = (unsigned __int64)v61;
    if ( v30 > 0xF )
    {
      na = v30;
      srcb = v31;
      v42 = (char *)sub_22409D0((__int64)&v59, &v55, 0);
      v31 = srcb;
      v32 = na;
      v59 = (unsigned __int64)v42;
      v43 = v42;
      v61[0] = v55;
    }
    else
    {
      if ( v30 == 1 )
      {
        LOBYTE(v61[0]) = *v31;
        v34 = v61;
LABEL_53:
        v60 = v33;
        *((_BYTE *)v34 + v33) = 0;
        v35 = v60;
        v36 = (char *)v59;
LABEL_58:
        sub_2241490((unsigned __int64 *)&v56, v36, v35);
        if ( (_QWORD *)v59 != v61 )
          j_j___libc_free_0(v59);
        v39 = 0x3FFFFFFFFFFFFFFFLL - v57;
        switch ( v4 )
        {
          case 1:
            if ( v39 <= 0xC )
              goto LABEL_72;
            sub_2241490((unsigned __int64 *)&v56, " is a kernel,", 0xDu);
            break;
          case 2:
            if ( v39 <= 0x16 )
              goto LABEL_72;
            sub_2241490((unsigned __int64 *)&v56, " has an image argument,", 0x17u);
            break;
          default:
            break;
        }
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v57) <= 0x4A )
          goto LABEL_72;
        sub_2241490(
          (unsigned __int64 *)&v56,
          " so overriding noinline attribute. The function may be inlined when called.",
          0x4Bu);
        sub_CEB590(&v56, 1, v40, v41);
        if ( v56 != (_QWORD *)v58 )
          j_j___libc_free_0((unsigned __int64)v56);
        goto LABEL_18;
      }
      if ( !v30 )
      {
        v34 = v61;
        goto LABEL_53;
      }
      v43 = (char *)v61;
    }
    memcpy(v43, v31, v32);
    v33 = v55;
    v34 = (_QWORD *)v59;
    goto LABEL_53;
  }
  return v50;
}
