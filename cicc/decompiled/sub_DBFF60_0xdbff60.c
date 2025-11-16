// Function: sub_DBFF60
// Address: 0xdbff60
//
_QWORD *__fastcall sub_DBFF60(__int64 a1, unsigned int *a2, __int64 a3, unsigned int a4)
{
  __int64 **v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rdx
  _QWORD *result; // rax
  unsigned int v11; // eax
  __int64 v12; // r9
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  _QWORD *v18; // rax
  unsigned int v19; // ecx
  unsigned int v20; // edx
  __int64 v21; // rcx
  void *v22; // r9
  size_t v23; // r15
  __int64 v24; // r8
  size_t v25; // r8
  __int64 *v26; // r15
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 *v30; // r14
  __int64 *v31; // rax
  __int64 **v32; // r15
  __int64 *v33; // rbx
  __int64 v34; // rsi
  _QWORD *v35; // rax
  __int64 *v36; // rdi
  signed __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 *v40; // r15
  __int64 v41; // rax
  __int64 *v42; // rcx
  __int64 v43; // rax
  __int64 *v44; // rbx
  __int64 v45; // rax
  __int64 *v46; // r12
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 *v49; // r15
  __int64 *v50; // r15
  __int64 *v51; // r15
  __int64 v52; // rax
  signed __int64 v53; // rax
  __int64 v54; // rax
  __int64 *v55; // r15
  __int64 v56; // rax
  __int64 *v57; // r15
  __int64 v58; // rax
  __int64 *v59; // r15
  void *src; // [rsp+8h] [rbp-88h]
  void *srcb; // [rsp+8h] [rbp-88h]
  __int64 **srca; // [rsp+8h] [rbp-88h]
  __int64 *v63; // [rsp+10h] [rbp-80h]
  unsigned __int64 v64; // [rsp+10h] [rbp-80h]
  __int64 v65; // [rsp+10h] [rbp-80h]
  __int64 *v66; // [rsp+10h] [rbp-80h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  unsigned int v68; // [rsp+28h] [rbp-68h]
  _QWORD *v69; // [rsp+28h] [rbp-68h]
  __int64 *v70; // [rsp+30h] [rbp-60h] BYREF
  __int64 v71; // [rsp+38h] [rbp-58h]
  _BYTE v72[80]; // [rsp+40h] [rbp-50h] BYREF

  v7 = (__int64 **)a2;
  while ( 1 )
  {
    v8 = a2[2];
    v9 = *(__int64 **)a2;
    if ( v8 == 1 )
      return (_QWORD *)*v9;
    if ( !sub_D968A0(v9[v8 - 1]) )
      break;
    --a2[2];
    a4 = 0;
  }
  v11 = sub_DBF900(a1, 8, *(__int64 **)a2, a2[2], a4);
  v13 = *(__int64 **)a2;
  v68 = v11;
  v14 = **v7;
  v67 = v14;
  if ( *(_WORD *)(v14 + 24) != 8 )
    return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
  v15 = *(_QWORD *)(v14 + 48);
  if ( a3 == v15 )
  {
LABEL_13:
    v17 = *(_QWORD **)a3;
    v18 = *(_QWORD **)v15;
    if ( *(_QWORD *)a3 )
    {
      v19 = 1;
      do
      {
        v17 = (_QWORD *)*v17;
        ++v19;
      }
      while ( v17 );
      if ( !v18 )
      {
        v20 = 1;
        goto LABEL_19;
      }
    }
    else
    {
      if ( !v18 )
        return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
      v19 = 1;
    }
    v20 = 1;
    do
    {
      v18 = (_QWORD *)*v18;
      ++v20;
    }
    while ( v18 );
LABEL_19:
    if ( v19 >= v20 )
      return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
    goto LABEL_20;
  }
  if ( v15 )
  {
    v16 = *(_QWORD **)(v14 + 48);
    do
    {
      v16 = (_QWORD *)*v16;
      if ( (_QWORD *)a3 == v16 )
        goto LABEL_13;
    }
    while ( v16 );
  }
  if ( v15 == a3 )
    return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
  if ( a3 )
  {
    v35 = (_QWORD *)a3;
    do
    {
      v35 = (_QWORD *)*v35;
      if ( (_QWORD *)v15 == v35 )
        return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
    }
    while ( v35 );
  }
  if ( !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 40), **(_QWORD **)(a3 + 32), **(_QWORD **)(v15 + 32)) )
    goto LABEL_33;
LABEL_20:
  v21 = *(_QWORD *)(v67 + 40);
  v22 = *(void **)(v67 + 32);
  v23 = 8 * v21;
  v70 = (__int64 *)v72;
  v24 = (8 * v21) >> 3;
  v71 = 0x400000000LL;
  if ( (unsigned __int64)(8 * v21) > 0x20 )
  {
    v64 = (8 * v21) >> 3;
    srcb = v22;
    sub_C8D5F0((__int64)&v70, v72, v64, 8u, v24, (__int64)v22);
    v24 = v64;
    v22 = srcb;
    v36 = &v70[(unsigned int)v71];
  }
  else
  {
    if ( !v23 )
      goto LABEL_22;
    v36 = (__int64 *)v72;
  }
  v65 = v24;
  memcpy(v36, v22, v23);
  v23 = (unsigned int)v71;
  v24 = v65;
LABEL_22:
  v25 = v23 + v24;
  LODWORD(v71) = v25;
  **v7 = **(_QWORD **)(v67 + 32);
  v26 = *v7;
  v27 = *((unsigned int *)v7 + 2);
  v28 = (__int64)&(*v7)[v27];
  v29 = (v27 * 8) >> 5;
  v63 = (__int64 *)v28;
  if ( !v29 )
  {
LABEL_46:
    v37 = (char *)v63 - (char *)v26;
    if ( (char *)v63 - (char *)v26 != 16 )
    {
      if ( v37 != 24 )
      {
        if ( v37 != 8 )
          goto LABEL_50;
        goto LABEL_49;
      }
      v34 = *v26;
      if ( !sub_DADE90(a1, *v26, a3) )
        goto LABEL_30;
      ++v26;
    }
    v34 = *v26;
    if ( !sub_DADE90(a1, *v26, a3) )
      goto LABEL_30;
    ++v26;
LABEL_49:
    v34 = *v26;
    if ( sub_DADE90(a1, *v26, a3) )
      goto LABEL_50;
    goto LABEL_30;
  }
  src = (void *)v15;
  v30 = *v7;
  v31 = &v26[4 * v29];
  v32 = v7;
  v33 = v31;
  while ( 1 )
  {
    v34 = *v30;
    if ( !sub_DADE90(a1, *v30, a3) )
    {
      v7 = v32;
      v26 = v30;
      v15 = (__int64)src;
      goto LABEL_30;
    }
    v34 = v30[1];
    if ( !sub_DADE90(a1, v34, a3) )
    {
      v7 = v32;
      v49 = v30;
      v15 = (__int64)src;
      v26 = v49 + 1;
      goto LABEL_30;
    }
    v34 = v30[2];
    if ( !sub_DADE90(a1, v34, a3) )
    {
      v7 = v32;
      v50 = v30;
      v15 = (__int64)src;
      v26 = v50 + 2;
      goto LABEL_30;
    }
    v34 = v30[3];
    if ( !sub_DADE90(a1, v34, a3) )
      break;
    v30 += 4;
    if ( v33 == v30 )
    {
      v7 = v32;
      v26 = v30;
      v15 = (__int64)src;
      goto LABEL_46;
    }
  }
  v7 = v32;
  v51 = v30;
  v15 = (__int64)src;
  v26 = v51 + 3;
LABEL_30:
  if ( v63 != v26 )
    goto LABEL_31;
LABEL_50:
  v38 = sub_DBFF60(a1, v7, a3, (unsigned __int8)v68 & (*(_WORD *)(v67 + 28) & 6 | 1u), v25);
  *v70 = v38;
  v40 = v70;
  v41 = (unsigned int)v71;
  v42 = &v70[v41];
  v43 = (v41 * 8) >> 5;
  v66 = v42;
  if ( !v43 )
  {
LABEL_73:
    v53 = (char *)v66 - (char *)v40;
    if ( (char *)v66 - (char *)v40 != 16 )
    {
      if ( v53 != 24 )
      {
        if ( v53 != 8 )
          goto LABEL_59;
        goto LABEL_76;
      }
      v34 = *v40;
      if ( !sub_DADE90(a1, *v40, v15) )
        goto LABEL_58;
      ++v40;
    }
    v34 = *v40;
    if ( !sub_DADE90(a1, *v40, v15) )
      goto LABEL_58;
    ++v40;
LABEL_76:
    v34 = *v40;
    if ( sub_DADE90(a1, *v40, v15) )
      goto LABEL_59;
    goto LABEL_58;
  }
  srca = v7;
  v44 = &v70[4 * v43];
  v45 = a3;
  v46 = v70;
  v47 = v45;
  while ( 1 )
  {
    v34 = *v46;
    if ( !sub_DADE90(a1, *v46, v15) )
    {
      v48 = v47;
      v7 = srca;
      v40 = v46;
      a3 = v48;
      goto LABEL_58;
    }
    v34 = v46[1];
    if ( !sub_DADE90(a1, v34, v15) )
    {
      v56 = v47;
      v57 = v46;
      v7 = srca;
      a3 = v56;
      v40 = v57 + 1;
      goto LABEL_58;
    }
    v34 = v46[2];
    if ( !sub_DADE90(a1, v34, v15) )
    {
      v54 = v47;
      v55 = v46;
      v7 = srca;
      a3 = v54;
      v40 = v55 + 2;
      goto LABEL_58;
    }
    v34 = v46[3];
    if ( !sub_DADE90(a1, v34, v15) )
      break;
    v46 += 4;
    if ( v44 == v46 )
    {
      v52 = v47;
      v7 = srca;
      v40 = v46;
      a3 = v52;
      goto LABEL_73;
    }
  }
  v58 = v47;
  v59 = v46;
  v7 = srca;
  a3 = v58;
  v40 = v59 + 3;
LABEL_58:
  if ( v66 != v40 )
  {
LABEL_31:
    **v7 = v67;
    if ( v70 != (__int64 *)v72 )
      _libc_free(v70, v34);
LABEL_33:
    v13 = *v7;
    return sub_DAFB70(a1, (unsigned __int64 *)v13, *((unsigned int *)v7 + 2), a3, v68, v12);
  }
LABEL_59:
  result = (_QWORD *)sub_DBFF60(a1, &v70, v15, ((unsigned __int8)v68 | 1) & *(_WORD *)(v67 + 28) & 7u, v39);
  if ( v70 != (__int64 *)v72 )
  {
    v69 = result;
    _libc_free(v70, &v70);
    return v69;
  }
  return result;
}
