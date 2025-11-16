// Function: sub_2F97470
// Address: 0x2f97470
//
__int64 *__fastcall sub_2F97470(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 *result; // rax
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned int *v16; // r15
  __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // rdi
  int v20; // eax
  __int64 *v21; // r15
  __int64 *v22; // rdi
  __int64 *v23; // r12
  __int64 v24; // r14
  _QWORD *v25; // rbx
  _QWORD *v26; // r14
  _QWORD *v27; // r15
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax
  __int64 v30; // r15
  char v31; // dl
  __int64 v32; // r9
  int v33; // esi
  unsigned int v34; // edi
  _QWORD *v35; // rax
  __int64 v36; // r10
  _DWORD *v37; // rax
  _QWORD *v38; // rax
  int v39; // ecx
  __int64 v40; // rdi
  int v41; // ecx
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 *v44; // rax
  __int64 v45; // r8
  unsigned int v46; // eax
  __int64 v47; // rdx
  __int64 *v48; // r14
  __int64 *v49; // r14
  __int64 *v50; // rbx
  __int64 *v51; // r15
  unsigned __int64 v52; // rdi
  __int64 v53; // r12
  __int64 v54; // r12
  int v55; // ecx
  unsigned int v56; // esi
  unsigned int v57; // eax
  _QWORD *v58; // r8
  int v59; // ecx
  unsigned int v60; // edi
  __int64 v61; // rax
  int v62; // r11d
  __int64 v63; // r9
  int v64; // esi
  unsigned int v65; // eax
  __int64 v66; // rcx
  __int64 v67; // r9
  int v68; // edx
  unsigned int v69; // eax
  __int64 v70; // r11
  int v71; // esi
  _QWORD *v72; // rcx
  int v73; // esi
  int v74; // edx
  int v75; // eax
  int v76; // r9d
  int v77; // r10d
  _QWORD *v78; // rdi
  __int64 *v79; // [rsp+0h] [rbp-50h]
  __int64 *v80; // [rsp+8h] [rbp-48h]
  __int64 v81; // [rsp+10h] [rbp-40h] BYREF
  __int64 v82; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v8 = 32LL * *(unsigned int *)(a2 + 88);
  result = (__int64 *)(v7 + v8);
  v79 = (__int64 *)(v7 + v8);
  if ( v7 == v7 + v8 )
    goto LABEL_87;
  v11 = (__int64 *)(v7 + 8);
  while ( 2 )
  {
    v12 = (__int64 *)*v11;
    if ( (__int64 *)*v11 == v11 )
      goto LABEL_25;
    while ( 1 )
    {
      v15 = *(_QWORD *)(a1 + 2904);
      v16 = (unsigned int *)v12[2];
      v13 = *(unsigned int *)(v15 + 200);
      if ( v16[50] <= (unsigned int)v13 )
        break;
      v82 = 0;
      v81 = v15 | 6;
      v17 = *(_QWORD *)v15;
      if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v15 + 68LL) - 1 <= 1
        && (*(_BYTE *)(*(_QWORD *)(v17 + 32) + 64LL) & 0x10) != 0 )
      {
LABEL_14:
        v19 = *(_QWORD *)v16;
        if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v16 + 68LL) - 1 > 1
          || (LODWORD(v14) = 1, (*(_BYTE *)(*(_QWORD *)(v19 + 32) + 64LL) & 8) == 0) )
        {
          v20 = *(_DWORD *)(v19 + 44);
          if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
            v14 = (*(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL) >> 19) & 1LL;
          else
            LOBYTE(v14) = sub_2E88A90(v19, 0x80000, 1);
          LODWORD(v14) = (unsigned __int8)v14;
        }
        goto LABEL_8;
      }
      v18 = *(_DWORD *)(v17 + 44);
      if ( (v18 & 4) != 0 || (v18 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v17 + 16) + 24LL) & 0x100000LL) != 0 )
          goto LABEL_14;
      }
      else if ( sub_2E88A90(v17, 0x100000, 1) )
      {
        goto LABEL_14;
      }
      LODWORD(v14) = 0;
LABEL_8:
      HIDWORD(v82) = v14;
      sub_2F8F1B0((__int64)v16, (__int64)&v81, 1u, v13, a5, a6);
      v12 = (__int64 *)*v12;
      if ( v12 == v11 )
        goto LABEL_23;
    }
    if ( v16 == (unsigned int *)v15 )
      v12 = (__int64 *)*v12;
LABEL_23:
    v21 = (__int64 *)*v11;
    while ( v12 != v21 )
    {
      v22 = v21;
      v21 = (__int64 *)*v21;
      --v11[2];
      sub_2208CA0(v22);
      j_j___libc_free_0((unsigned __int64)v22);
    }
LABEL_25:
    if ( v79 != v11 + 3 )
    {
      v11 += 4;
      continue;
    }
    break;
  }
  v23 = *(__int64 **)(a2 + 80);
  v24 = 4LL * *(unsigned int *)(a2 + 88);
  result = &v23[v24];
  v80 = &v23[v24];
  if ( v23 == &v23[v24] )
  {
LABEL_87:
    *(_DWORD *)(a2 + 88) = 0;
    *(_DWORD *)(a2 + 224) = 0;
    return result;
  }
  v25 = v23 + 1;
  if ( (_QWORD *)*v25 == v25 )
    goto LABEL_42;
LABEL_29:
  if ( v23 == v25 - 1 )
    goto LABEL_39;
  v26 = (_QWORD *)v23[1];
  v27 = v23 + 1;
  *v23 = *(v25 - 1);
  while ( v26 != v27 )
  {
    v28 = (unsigned __int64)v26;
    v26 = (_QWORD *)*v26;
    j_j___libc_free_0(v28);
  }
  if ( (_QWORD *)*v25 == v25 )
  {
    v23[2] = (__int64)v27;
    v23[1] = (__int64)v27;
    v23[3] = 0;
  }
  else
  {
    v23[1] = *v25;
    v29 = (_QWORD *)v25[1];
    v23[2] = (__int64)v29;
    *v29 = v27;
    *(_QWORD *)(v23[1] + 8) = v27;
    v23[3] = v25[2];
    v25[1] = v25;
    *v25 = v25;
    v25[2] = 0;
  }
  v30 = ((__int64)v23 - *(_QWORD *)(a2 + 80)) >> 5;
  v31 = *(_BYTE *)(a2 + 8) & 1;
  if ( v31 )
  {
    v32 = a2 + 16;
    v33 = 3;
    goto LABEL_36;
  }
  v56 = *(_DWORD *)(a2 + 24);
  v32 = *(_QWORD *)(a2 + 16);
  if ( !v56 )
  {
    v57 = *(_DWORD *)(a2 + 8);
    ++*(_QWORD *)a2;
    v58 = 0;
    v59 = (v57 >> 1) + 1;
    goto LABEL_61;
  }
  v33 = v56 - 1;
LABEL_36:
  v34 = v33 & (37 * *v23);
  v35 = (_QWORD *)(v32 + 16LL * v34);
  v36 = *v35;
  if ( *v23 == *v35 )
  {
LABEL_37:
    v37 = v35 + 1;
    goto LABEL_38;
  }
  v62 = 1;
  v58 = 0;
  while ( v36 != -4096 )
  {
    if ( v36 == -8192 && !v58 )
      v58 = v35;
    v34 = v33 & (v62 + v34);
    v35 = (_QWORD *)(v32 + 16LL * v34);
    v36 = *v35;
    if ( *v23 == *v35 )
      goto LABEL_37;
    ++v62;
  }
  v60 = 12;
  v56 = 4;
  if ( !v58 )
    v58 = v35;
  v57 = *(_DWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v59 = (v57 >> 1) + 1;
  if ( !v31 )
  {
    v56 = *(_DWORD *)(a2 + 24);
LABEL_61:
    v60 = 3 * v56;
  }
  if ( 4 * v59 >= v60 )
  {
    sub_2F96BB0(a2, 2 * v56);
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v63 = a2 + 16;
      v64 = 3;
    }
    else
    {
      v73 = *(_DWORD *)(a2 + 24);
      v63 = *(_QWORD *)(a2 + 16);
      if ( !v73 )
        goto LABEL_113;
      v64 = v73 - 1;
    }
    v65 = v64 & (37 * *v23);
    v58 = (_QWORD *)(v63 + 16LL * v65);
    v66 = *v58;
    if ( *v23 != *v58 )
    {
      v77 = 1;
      v78 = 0;
      while ( v66 != -4096 )
      {
        if ( v66 == -8192 && !v78 )
          v78 = v58;
        v65 = v64 & (v77 + v65);
        v58 = (_QWORD *)(v63 + 16LL * v65);
        v66 = *v58;
        if ( *v23 == *v58 )
          goto LABEL_77;
        ++v77;
      }
      if ( v78 )
      {
        v57 = *(_DWORD *)(a2 + 8);
        v58 = v78;
        goto LABEL_64;
      }
    }
    goto LABEL_77;
  }
  if ( v56 - *(_DWORD *)(a2 + 12) - v59 <= v56 >> 3 )
  {
    sub_2F96BB0(a2, v56);
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v67 = a2 + 16;
      v68 = 3;
      goto LABEL_80;
    }
    v74 = *(_DWORD *)(a2 + 24);
    v67 = *(_QWORD *)(a2 + 16);
    if ( v74 )
    {
      v68 = v74 - 1;
LABEL_80:
      v69 = v68 & (37 * *v23);
      v58 = (_QWORD *)(v67 + 16LL * v69);
      v70 = *v58;
      if ( *v23 != *v58 )
      {
        v71 = 1;
        v72 = 0;
        while ( v70 != -4096 )
        {
          if ( !v72 && v70 == -8192 )
            v72 = v58;
          v69 = v68 & (v71 + v69);
          v58 = (_QWORD *)(v67 + 16LL * v69);
          v70 = *v58;
          if ( *v23 == *v58 )
            goto LABEL_77;
          ++v71;
        }
        if ( v72 )
          v58 = v72;
      }
LABEL_77:
      v57 = *(_DWORD *)(a2 + 8);
      goto LABEL_64;
    }
LABEL_113:
    *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
    BUG();
  }
LABEL_64:
  *(_DWORD *)(a2 + 8) = (2 * (v57 >> 1) + 2) | v57 & 1;
  if ( *v58 != -4096 )
    --*(_DWORD *)(a2 + 12);
  v61 = *v23;
  *((_DWORD *)v58 + 2) = 0;
  *v58 = v61;
  v37 = v58 + 1;
LABEL_38:
  *v37 = v30;
LABEL_39:
  v23 += 4;
LABEL_40:
  while ( 1 )
  {
    v38 = v25 + 4;
    if ( v80 == v25 + 3 )
      break;
    while ( 1 )
    {
      v25 = v38;
      if ( (_QWORD *)*v25 != v25 )
        goto LABEL_29;
LABEL_42:
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v40 = a2 + 16;
        v41 = 3;
      }
      else
      {
        v39 = *(_DWORD *)(a2 + 24);
        v40 = *(_QWORD *)(a2 + 16);
        if ( !v39 )
          goto LABEL_40;
        v41 = v39 - 1;
      }
      v42 = *(v25 - 1);
      v43 = v41 & (37 * v42);
      v44 = (__int64 *)(v40 + 16LL * v43);
      v45 = *v44;
      if ( v42 != *v44 )
        break;
LABEL_46:
      *v44 = -8192;
      v46 = *(_DWORD *)(a2 + 8);
      ++*(_DWORD *)(a2 + 12);
      *(_DWORD *)(a2 + 8) = (2 * (v46 >> 1) - 2) | v46 & 1;
      v38 = v25 + 4;
      if ( v80 == v25 + 3 )
        goto LABEL_47;
    }
    v75 = 1;
    while ( v45 != -4096 )
    {
      v76 = v75 + 1;
      v43 = v41 & (v75 + v43);
      v44 = (__int64 *)(v40 + 16LL * v43);
      v45 = *v44;
      if ( v42 == *v44 )
        goto LABEL_46;
      v75 = v76;
    }
  }
LABEL_47:
  v47 = *(_QWORD *)(a2 + 80);
  v48 = (__int64 *)(v47 + 32LL * *(unsigned int *)(a2 + 88));
  if ( v48 != v23 )
  {
    v49 = v48 - 3;
    do
    {
      v50 = (__int64 *)*v49;
      v51 = v49 - 1;
      while ( v50 != v49 )
      {
        v52 = (unsigned __int64)v50;
        v50 = (__int64 *)*v50;
        j_j___libc_free_0(v52);
      }
      v49 -= 4;
    }
    while ( v23 != v51 );
    v47 = *(_QWORD *)(a2 + 80);
  }
  *(_DWORD *)(a2 + 224) = 0;
  v53 = ((__int64)v23 - v47) >> 5;
  *(_DWORD *)(a2 + 88) = v53;
  v54 = 32LL * (unsigned int)v53;
  result = (__int64 *)(v47 + v54);
  if ( v47 + v54 != v47 )
  {
    v55 = 0;
    do
    {
      v55 += *(_DWORD *)(v47 + 24);
      v47 += 32;
      *(_DWORD *)(a2 + 224) = v55;
    }
    while ( result != (__int64 *)v47 );
  }
  return result;
}
