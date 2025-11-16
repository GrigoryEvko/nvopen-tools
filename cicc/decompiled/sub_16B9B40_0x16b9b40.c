// Function: sub_16B9B40
// Address: 0x16b9b40
//
__int64 __fastcall sub_16B9B40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 **v5; // r14
  __int64 v6; // rax
  unsigned __int64 **i; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // rsi
  unsigned __int64 **v13; // rbx
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // r15
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // r8
  unsigned __int64 *v19; // rsi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 *v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // r10
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // r9
  unsigned __int64 *v29; // rsi
  unsigned __int64 v30; // rdx
  _QWORD *v31; // r9
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // rax
  char *v34; // rsi
  unsigned __int64 *v35; // r12
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // r15
  unsigned __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // r9
  _WORD *v44; // rdi
  const char *v45; // rsi
  size_t v46; // rdx
  unsigned __int64 v47; // rax
  __int64 v48; // r9
  void *v49; // rdi
  const char *v50; // rsi
  size_t v51; // rdx
  __int64 v52; // rdx
  __int64 **v53; // r15
  __int64 **v54; // r14
  __int64 result; // rax
  _BYTE *v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+8h] [rbp-C8h]
  __int64 v61; // [rsp+10h] [rbp-C0h]
  __int64 v62; // [rsp+10h] [rbp-C0h]
  __int64 v64; // [rsp+28h] [rbp-A8h]
  __int64 v65; // [rsp+28h] [rbp-A8h]
  unsigned __int64 *v66; // [rsp+28h] [rbp-A8h]
  __int64 v68; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 *v69; // [rsp+48h] [rbp-88h] BYREF
  void *base; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 *v71; // [rsp+58h] [rbp-78h]
  unsigned __int64 *v72; // [rsp+60h] [rbp-70h]
  __int64 v73; // [rsp+70h] [rbp-60h] BYREF
  int v74; // [rsp+78h] [rbp-58h] BYREF
  unsigned __int64 *v75; // [rsp+80h] [rbp-50h]
  int *v76; // [rsp+88h] [rbp-48h]
  int *v77; // [rsp+90h] [rbp-40h]
  __int64 v78; // [rsp+98h] [rbp-38h]

  base = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0;
  v75 = 0;
  v76 = &v74;
  v77 = &v74;
  v78 = 0;
  v4 = sub_16B0440(a1, a2);
  v5 = *(unsigned __int64 ***)(v4 + 88);
  if ( v5 == *(unsigned __int64 ***)(v4 + 80) )
    v6 = *(unsigned int *)(v4 + 100);
  else
    v6 = *(unsigned int *)(v4 + 96);
  for ( i = &v5[v6]; i != v5; ++v5 )
  {
    if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  v8 = sub_16B0440(a1, a2);
  v9 = *(_QWORD *)(v8 + 88);
  if ( v9 == *(_QWORD *)(v8 + 80) )
    v10 = *(unsigned int *)(v8 + 100);
  else
    v10 = *(unsigned int *)(v8 + 96);
  v11 = v71;
  if ( v5 != (unsigned __int64 **)(v9 + 8 * v10) )
  {
    v12 = v71;
    v13 = (unsigned __int64 **)(v9 + 8 * v10);
    do
    {
      v14 = *v5;
      v69 = *v5;
      if ( v72 == v12 )
      {
        sub_16B9260((__int64)&base, v12, &v69);
        v12 = v71;
      }
      else
      {
        if ( v12 )
        {
          *v12 = (unsigned __int64)v14;
          v12 = v71;
        }
        v71 = ++v12;
      }
      for ( ++v5; i != v5; ++v5 )
      {
        if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
    }
    while ( v13 != v5 );
    v11 = v12;
  }
  v15 = (unsigned __int64 *)base;
  if ( (char *)v11 - (_BYTE *)base > 8 )
  {
    qsort(base, ((char *)v11 - (_BYTE *)base) >> 3, 8u, (__compar_fn_t)sub_16B0140);
    v11 = v71;
    v15 = (unsigned __int64 *)base;
  }
  if ( v11 != v15 )
  {
    v64 = a3;
    v16 = v15;
    while ( 1 )
    {
      v17 = v75;
      if ( v75 )
      {
        v18 = *v16;
        v19 = (unsigned __int64 *)&v74;
        do
        {
          while ( 1 )
          {
            v20 = v17[2];
            v21 = v17[3];
            if ( v17[4] >= v18 )
              break;
            v17 = (unsigned __int64 *)v17[3];
            if ( !v21 )
              goto LABEL_28;
          }
          v19 = v17;
          v17 = (unsigned __int64 *)v17[2];
        }
        while ( v20 );
LABEL_28:
        if ( v19 != (unsigned __int64 *)&v74 && v19[4] <= v18 )
          goto LABEL_31;
      }
      else
      {
        v19 = (unsigned __int64 *)&v74;
      }
      v69 = v16;
      v19 = sub_16B9A70(&v73, v19, &v69);
LABEL_31:
      v22 = (__int64 *)v19[5];
      v23 = v19[7];
      v19[5] = 0;
      v19[6] = 0;
      v19[7] = 0;
      if ( v22 )
        j_j___libc_free_0(v22, v23 - (_QWORD)v22);
      if ( v11 == ++v16 )
      {
        a3 = v64;
        v24 = *(unsigned int *)(a2 + 8);
        if ( !*(_DWORD *)(a2 + 8) )
          goto LABEL_50;
        goto LABEL_35;
      }
    }
  }
  v24 = *(unsigned int *)(a2 + 8);
  if ( !*(_DWORD *)(a2 + 8) )
    goto LABEL_77;
LABEL_35:
  v65 = a3;
  v25 = 0;
  do
  {
    v26 = *(_QWORD *)(*(_QWORD *)a2 + 16 * v25 + 8);
    v27 = v75;
    v68 = v26;
    if ( !v75 )
    {
      v29 = (unsigned __int64 *)&v74;
LABEL_43:
      v22 = &v73;
      v69 = (unsigned __int64 *)(v26 + 72);
      v29 = sub_16B9A70(&v73, v29, &v69);
      goto LABEL_44;
    }
    v28 = *(_QWORD *)(v26 + 72);
    v29 = (unsigned __int64 *)&v74;
    do
    {
      while ( 1 )
      {
        v22 = (__int64 *)v27[2];
        v30 = v27[3];
        if ( v27[4] >= v28 )
          break;
        v27 = (unsigned __int64 *)v27[3];
        if ( !v30 )
          goto LABEL_41;
      }
      v29 = v27;
      v27 = (unsigned __int64 *)v27[2];
    }
    while ( v22 );
LABEL_41:
    if ( v29 == (unsigned __int64 *)&v74 || v29[4] > v28 )
      goto LABEL_43;
LABEL_44:
    v31 = (_QWORD *)v29[6];
    if ( v31 == (_QWORD *)v29[7] )
    {
      v22 = (__int64 *)(v29 + 5);
      sub_16B90D0((__int64)(v29 + 5), (_BYTE *)v29[6], &v68);
    }
    else
    {
      if ( v31 )
      {
        *v31 = v68;
        v31 = (_QWORD *)v29[6];
      }
      v29[6] = (unsigned __int64)(v31 + 1);
    }
    ++v25;
  }
  while ( v25 != v24 );
  a3 = v65;
LABEL_50:
  v32 = (unsigned __int64 *)base;
  v66 = v71;
  if ( v71 != base )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        v33 = v75;
        if ( !v75 )
        {
          v35 = (unsigned __int64 *)&v74;
          goto LABEL_58;
        }
        v34 = (char *)*v32;
        v35 = (unsigned __int64 *)&v74;
        do
        {
          while ( 1 )
          {
            v36 = v33[2];
            v37 = v33[3];
            if ( v33[4] >= (unsigned __int64)v34 )
              break;
            v33 = (unsigned __int64 *)v33[3];
            if ( !v37 )
              goto LABEL_56;
          }
          v35 = v33;
          v33 = (unsigned __int64 *)v33[2];
        }
        while ( v36 );
LABEL_56:
        if ( v35 == (unsigned __int64 *)&v74 || v35[4] > (unsigned __int64)v34 )
        {
LABEL_58:
          v22 = &v73;
          v34 = (char *)v35;
          v69 = v32;
          v35 = sub_16B9A70(&v73, v35, &v69);
        }
        v38 = v35[6];
        v39 = v35[5];
        if ( *(_BYTE *)(a1 + 8) != 1 && v38 == v39 )
          goto LABEL_76;
        v41 = sub_16E8C20(v22, v34, v37);
        v42 = *(_BYTE **)(v41 + 24);
        if ( *(_BYTE **)(v41 + 16) == v42 )
        {
          v34 = "\n";
          sub_16E7EE0(v41, "\n", 1);
        }
        else
        {
          *v42 = 10;
          ++*(_QWORD *)(v41 + 24);
        }
        v43 = sub_16E8C20(v41, v34, v40);
        v44 = *(_WORD **)(v43 + 24);
        v45 = *(const char **)*v32;
        v46 = *(_QWORD *)(*v32 + 8);
        v47 = *(_QWORD *)(v43 + 16) - (_QWORD)v44;
        if ( v46 > v47 )
        {
          v57 = sub_16E7EE0(v43, v45);
          v44 = *(_WORD **)(v57 + 24);
          v43 = v57;
          v47 = *(_QWORD *)(v57 + 16) - (_QWORD)v44;
        }
        else if ( v46 )
        {
          v59 = v43;
          v61 = *(_QWORD *)(*v32 + 8);
          memcpy(v44, v45, v46);
          v43 = v59;
          v58 = *(_QWORD *)(v59 + 16);
          v46 = *(_QWORD *)(v59 + 24) + v61;
          *(_QWORD *)(v59 + 24) = v46;
          v44 = (_WORD *)v46;
          v47 = v58 - v46;
        }
        if ( v47 <= 1 )
        {
          v45 = ":\n";
          v44 = (_WORD *)v43;
          sub_16E7EE0(v43, ":\n", 2);
        }
        else
        {
          *v44 = 2618;
          *(_QWORD *)(v43 + 24) += 2LL;
        }
        if ( *(_QWORD *)(*v32 + 24) )
        {
          v48 = sub_16E8C20(v44, v45, v46);
          v49 = *(void **)(v48 + 24);
          v50 = *(const char **)(*v32 + 16);
          v51 = *(_QWORD *)(*v32 + 24);
          if ( v51 > *(_QWORD *)(v48 + 16) - (_QWORD)v49 )
          {
            v48 = sub_16E7EE0(v48, v50);
          }
          else if ( v51 )
          {
            v60 = v48;
            v62 = *(_QWORD *)(*v32 + 24);
            memcpy(v49, v50, v51);
            v48 = v60;
            *(_QWORD *)(v60 + 24) += v62;
          }
          v45 = "\n\n";
          v22 = (__int64 *)v48;
          sub_1263B40(v48, "\n\n");
          goto LABEL_73;
        }
        v22 = (__int64 *)sub_16E8C20(v44, v45, v46);
        v56 = (_BYTE *)v22[3];
        if ( (_BYTE *)v22[2] == v56 )
        {
          v45 = "\n";
          sub_16E7EE0(v22, "\n", 1);
LABEL_73:
          if ( v38 != v39 )
            goto LABEL_74;
        }
        else
        {
          *v56 = 10;
          ++v22[3];
          if ( v38 != v39 )
          {
LABEL_74:
            v53 = (__int64 **)v35[5];
            v54 = (__int64 **)v35[6];
            while ( v54 != v53 )
            {
              v22 = *v53++;
              (*(void (__fastcall **)(__int64 *, __int64))(*v22 + 48))(v22, a3);
            }
LABEL_76:
            if ( v66 == ++v32 )
              goto LABEL_77;
            continue;
          }
        }
        break;
      }
      ++v32;
      v22 = (__int64 *)sub_16E8C20(v22, v45, v52);
      sub_1263B40((__int64)v22, "  This option category has no options.\n");
      if ( v66 == v32 )
        break;
      continue;
    }
  }
LABEL_77:
  result = sub_16B8E00((__int64)&v73, v75);
  if ( base )
    return j_j___libc_free_0(base, (char *)v72 - (_BYTE *)base);
  return result;
}
