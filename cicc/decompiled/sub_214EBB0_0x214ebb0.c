// Function: sub_214EBB0
// Address: 0x214ebb0
//
void __fastcall sub_214EBB0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r12
  const char *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  void *v13; // rdi
  __int64 v14; // r15
  void *v15; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // r12
  _BYTE *v18; // rax
  _BYTE *v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // r15
  __int64 v22; // rax
  size_t v23; // rdx
  _BYTE *v24; // rdi
  __int64 v25; // rcx
  _BYTE *v26; // rax
  size_t v27; // r14
  __int64 v28; // r10
  __int64 v29; // rax
  size_t v30; // rdx
  char *v31; // rsi
  size_t v32; // r14
  _BYTE *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 *v37; // rsi
  __int64 v38; // r14
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // rdi
  _BYTE *v44; // rax
  unsigned int v45; // eax
  __int64 *v46; // rsi
  __int64 v47; // rsi
  __int64 v48; // rdi
  _BYTE *v49; // rax
  unsigned int v50; // eax
  __int64 *v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+8h] [rbp-48h]
  __int64 v56; // [rsp+8h] [rbp-48h]
  char *src; // [rsp+10h] [rbp-40h]
  void *srca; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+18h] [rbp-38h]
  int na; // [rsp+18h] [rbp-38h]

  v4 = (__int64)(a4[1] - *a4) >> 3;
  if ( (_DWORD)v4 )
  {
    v7 = sub_1649960(*(_QWORD *)(*(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8)) + 136LL));
    n = v8;
    src = (char *)v7;
    v9 = sub_1263B40(a2, ".metadata ");
    v10 = *(unsigned int *)(a1 + 912);
    *(_DWORD *)(a1 + 912) = v10 + 1;
    v11 = sub_16E7A90(v9, v10);
    sub_1263B40(v11, " {\n");
    sub_1263B40(a2, "\t\"cl_kernel_attributes\",\n");
    v12 = sub_1263B40(a2, "\t\"");
    v13 = *(void **)(v12 + 24);
    v14 = v12;
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 < n )
    {
      v14 = sub_16E7EE0(v12, src, n);
    }
    else if ( n )
    {
      memcpy(v13, src, n);
      *(_QWORD *)(v14 + 24) += n;
    }
    v15 = 0;
    sub_1263B40(v14, "\",\n");
    sub_1263B40(a2, "\t\"");
    na = v4 - 1;
    v16 = (unsigned int)v4;
    v17 = a2;
    srca = (void *)v16;
    while ( 1 )
    {
      v21 = *(_QWORD *)(*a4 + 8LL * (_QWORD)v15);
      v22 = sub_161E970(*(_QWORD *)(v21 - 8LL * *(unsigned int *)(v21 + 8)));
      v24 = *(_BYTE **)(v17 + 24);
      v25 = v22;
      v26 = *(_BYTE **)(v17 + 16);
      v27 = v23;
      if ( v26 - v24 < v23 )
      {
        v54 = v25;
        v34 = sub_16E7EE0(v17, (char *)v25, v23);
        v25 = v54;
        v28 = v34;
        v26 = *(_BYTE **)(v34 + 16);
        v24 = *(_BYTE **)(v28 + 24);
      }
      else
      {
        v28 = v17;
        if ( v23 )
        {
          v56 = v25;
          memcpy(v24, (const void *)v25, v23);
          v26 = *(_BYTE **)(v17 + 16);
          v28 = v17;
          v25 = v56;
          v24 = (_BYTE *)(v27 + *(_QWORD *)(v17 + 24));
          *(_QWORD *)(v17 + 24) = v24;
        }
      }
      if ( v26 == v24 )
      {
        v53 = v25;
        sub_16E7EE0(v28, "(", 1u);
        v25 = v53;
      }
      else
      {
        *v24 = 40;
        ++*(_QWORD *)(v28 + 24);
      }
      if ( v27 != 13 )
        break;
      if ( *(_QWORD *)v25 != 0x657079745F636576LL || *(_DWORD *)(v25 + 8) != 1852401759 || *(_BYTE *)(v25 + 12) != 116 )
        goto LABEL_7;
      v29 = sub_161E970(*(_QWORD *)(v21 + 8 * (1LL - *(unsigned int *)(v21 + 8))));
      v19 = *(_BYTE **)(v17 + 24);
      v31 = (char *)v29;
      v18 = *(_BYTE **)(v17 + 16);
      v32 = v30;
      if ( v18 - v19 < v30 )
      {
        sub_16E7EE0(v17, v31, v30);
        v18 = *(_BYTE **)(v17 + 16);
        v19 = *(_BYTE **)(v17 + 24);
        goto LABEL_8;
      }
      if ( !v30 )
        goto LABEL_8;
      memcpy(v19, v31, v30);
      v33 = *(_BYTE **)(v17 + 16);
      v19 = (_BYTE *)(v32 + *(_QWORD *)(v17 + 24));
      *(_QWORD *)(v17 + 24) = v19;
      if ( v19 == v33 )
      {
LABEL_26:
        sub_16E7EE0(v17, ")", 1u);
        goto LABEL_10;
      }
LABEL_9:
      *v19 = 41;
      ++*(_QWORD *)(v17 + 24);
LABEL_10:
      if ( na == (_DWORD)v15 )
      {
LABEL_13:
        v15 = (char *)v15 + 1;
        if ( srca == v15 )
          goto LABEL_45;
      }
      else
      {
        v20 = *(_BYTE **)(v17 + 24);
        if ( *(_BYTE **)(v17 + 16) != v20 )
        {
          *v20 = 32;
          ++*(_QWORD *)(v17 + 24);
          goto LABEL_13;
        }
        v15 = (char *)v15 + 1;
        sub_16E7EE0(v17, " ", 1u);
        if ( srca == v15 )
        {
LABEL_45:
          sub_1263B40(v17, "\"\n}\n\n");
          return;
        }
      }
    }
    if ( v27 == 20
      && (!(*(_QWORD *)v25 ^ 0x6F72675F6B726F77LL | *(_QWORD *)(v25 + 8) ^ 0x5F657A69735F7075LL)
       && *(_DWORD *)(v25 + 16) == 1953393000
       || !(*(_QWORD *)v25 ^ 0x726F775F64716572LL | *(_QWORD *)(v25 + 8) ^ 0x5F70756F72675F6BLL)
       && *(_DWORD *)(v25 + 16) == 1702521203) )
    {
      v35 = *(unsigned int *)(v21 + 8);
      v36 = *(_QWORD *)(*(_QWORD *)(v21 + 8 * (1 - v35)) + 136LL);
      v37 = *(__int64 **)(v36 + 24);
      v38 = *(_QWORD *)(*(_QWORD *)(v21 + 8 * (3 - v35)) + 136LL);
      v39 = *(_DWORD *)(v36 + 32);
      if ( v39 > 0x40 )
        v40 = *v37;
      else
        v40 = (__int64)((_QWORD)v37 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
      v55 = *(_QWORD *)(*(_QWORD *)(v21 + 8 * (2 - v35)) + 136LL);
      v41 = sub_16E7AB0(v17, v40);
      v42 = v55;
      v43 = v41;
      v44 = *(_BYTE **)(v41 + 24);
      if ( *(_BYTE **)(v43 + 16) == v44 )
      {
        sub_16E7EE0(v43, ",", 1u);
        v42 = v55;
      }
      else
      {
        *v44 = 44;
        ++*(_QWORD *)(v43 + 24);
      }
      v45 = *(_DWORD *)(v42 + 32);
      v46 = *(__int64 **)(v42 + 24);
      if ( v45 > 0x40 )
        v47 = *v46;
      else
        v47 = (__int64)((_QWORD)v46 << (64 - (unsigned __int8)v45)) >> (64 - (unsigned __int8)v45);
      v48 = sub_16E7AB0(v17, v47);
      v49 = *(_BYTE **)(v48 + 24);
      if ( *(_BYTE **)(v48 + 16) == v49 )
      {
        sub_16E7EE0(v48, ",", 1u);
      }
      else
      {
        *v49 = 44;
        ++*(_QWORD *)(v48 + 24);
      }
      v50 = *(_DWORD *)(v38 + 32);
      v51 = *(__int64 **)(v38 + 24);
      if ( v50 > 0x40 )
        v52 = *v51;
      else
        v52 = (__int64)((_QWORD)v51 << (64 - (unsigned __int8)v50)) >> (64 - (unsigned __int8)v50);
      sub_16E7AB0(v17, v52);
    }
LABEL_7:
    v18 = *(_BYTE **)(v17 + 16);
    v19 = *(_BYTE **)(v17 + 24);
LABEL_8:
    if ( v19 == v18 )
      goto LABEL_26;
    goto LABEL_9;
  }
}
