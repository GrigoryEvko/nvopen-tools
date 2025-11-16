// Function: sub_DA7540
// Address: 0xda7540
//
__int64 __fastcall sub_DA7540(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 *a5)
{
  __int64 v6; // r12
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 result; // rax
  int v15; // eax
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  size_t v19; // rax
  size_t v20; // r15
  int v21; // eax
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 *v24; // r15
  unsigned __int8 *v25; // r13
  unsigned __int8 v26; // al
  __int64 v27; // rax
  unsigned int v28; // esi
  __int64 v29; // r8
  _QWORD *v30; // rdx
  unsigned __int8 *v31; // rdi
  __int64 *v32; // rdx
  __int64 v33; // r13
  int v34; // r8d
  unsigned __int8 **v35; // rcx
  int v36; // edi
  int v37; // edi
  int v38; // r11d
  int v39; // r11d
  __int64 v40; // r8
  unsigned int v41; // r10d
  unsigned __int8 *v42; // rdx
  int v43; // esi
  unsigned __int8 **v44; // r9
  int v45; // r11d
  int v46; // r11d
  int v47; // esi
  __int64 v48; // r8
  unsigned int v49; // r10d
  unsigned __int8 *v50; // rdx
  unsigned int v51; // [rsp+8h] [rbp-78h]
  int v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+10h] [rbp-70h]
  __int64 v54; // [rsp+10h] [rbp-70h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 *v56; // [rsp+20h] [rbp-60h]
  __int64 *v57; // [rsp+28h] [rbp-58h]
  __int64 v58; // [rsp+30h] [rbp-50h]
  __int64 v59; // [rsp+38h] [rbp-48h]
  __int64 v62; // [rsp+48h] [rbp-38h]

  v6 = a1;
  v8 = *(_DWORD *)(a3 + 24);
  v9 = *(_QWORD *)(a3 + 8);
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = v10 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
    {
LABEL_3:
      result = v12[1];
      if ( result )
        return result;
    }
    else
    {
      v15 = 1;
      while ( v13 != -4096 )
      {
        v34 = v15 + 1;
        v11 = v10 & (v15 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v6 == *v12 )
          goto LABEL_3;
        v15 = v34;
      }
    }
  }
  v16 = *(_QWORD *)(v6 + 40);
  if ( *(_BYTE *)(a2 + 84) )
  {
    v17 = *(_QWORD **)(a2 + 64);
    v18 = &v17[*(unsigned int *)(a2 + 76)];
    if ( v17 == v18 )
      return 0;
    while ( v16 != *v17 )
    {
      if ( v18 == ++v17 )
        return 0;
    }
  }
  else if ( !sub_C8CA60(a2 + 56, v16) )
  {
    return 0;
  }
  result = 0;
  if ( *(_BYTE *)v6 == 84 )
    return result;
  if ( !sub_D90BC0((unsigned __int8 *)v6) || *(_BYTE *)v6 == 84 )
    return 0;
  v19 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(v6 + 4) & 0x7FFFFFF) == 0 )
  {
    v57 = 0;
    v56 = 0;
LABEL_27:
    v33 = (char *)v56 - (char *)v57;
    result = sub_97D230((unsigned __int8 *)v6, v57, v56 - v57, a4, a5, 0);
    if ( v57 )
      goto LABEL_28;
    return result;
  }
  v20 = v19;
  v55 = 8 * v19;
  v57 = (__int64 *)sub_22077B0(8 * v19);
  v56 = &v57[v20];
  memset(v57, 0, v20 * 8);
  v21 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
  if ( !v21 )
    goto LABEL_27;
  v58 = a2;
  v22 = v6;
  v23 = 0;
  v24 = v57;
  v59 = (__int64)&v57[(unsigned int)(v21 - 1) + 1];
  while ( (*(_BYTE *)(v22 + 7) & 0x40) != 0 )
  {
    v25 = *(unsigned __int8 **)(*(_QWORD *)(v22 - 8) + v23);
    v26 = *v25;
    if ( *v25 > 0x1Cu )
      goto LABEL_20;
LABEL_32:
    if ( v26 > 0x15u )
    {
      *v24 = 0;
      v33 = v55;
      result = 0;
      goto LABEL_28;
    }
    *v24 = (__int64)v25;
LABEL_25:
    ++v24;
    v23 += 32;
    if ( v24 == (__int64 *)v59 )
    {
      v6 = v22;
      goto LABEL_27;
    }
  }
  v25 = *(unsigned __int8 **)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF) + v23);
  v26 = *v25;
  if ( *v25 <= 0x1Cu )
    goto LABEL_32;
LABEL_20:
  v27 = sub_DA7540(v25, v58, a3, a4, a5);
  v28 = *(_DWORD *)(a3 + 24);
  if ( v28 )
  {
    v29 = (v28 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v30 = (_QWORD *)(*(_QWORD *)(a3 + 8) + 16 * v29);
    v31 = (unsigned __int8 *)*v30;
    if ( v25 == (unsigned __int8 *)*v30 )
    {
LABEL_22:
      v32 = v30 + 1;
      goto LABEL_23;
    }
    v52 = 1;
    v35 = 0;
    while ( v31 != (unsigned __int8 *)-4096LL )
    {
      if ( v31 == (unsigned __int8 *)-8192LL && !v35 )
        v35 = (unsigned __int8 **)v30;
      LODWORD(v29) = (v28 - 1) & (v52 + v29);
      v30 = (_QWORD *)(*(_QWORD *)(a3 + 8) + 16LL * (unsigned int)v29);
      v31 = (unsigned __int8 *)*v30;
      if ( v25 == (unsigned __int8 *)*v30 )
        goto LABEL_22;
      ++v52;
    }
    v36 = *(_DWORD *)(a3 + 16);
    if ( !v35 )
      v35 = (unsigned __int8 **)v30;
    ++*(_QWORD *)a3;
    v37 = v36 + 1;
    if ( 4 * v37 < 3 * v28 )
    {
      if ( v28 - *(_DWORD *)(a3 + 20) - v37 <= v28 >> 3 )
      {
        v51 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
        v54 = v27;
        sub_DA7360(a3, v28);
        v45 = *(_DWORD *)(a3 + 24);
        if ( !v45 )
        {
LABEL_74:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
        v46 = v45 - 1;
        v47 = 1;
        v48 = *(_QWORD *)(a3 + 8);
        v44 = 0;
        v37 = *(_DWORD *)(a3 + 16) + 1;
        v27 = v54;
        v49 = v46 & v51;
        v35 = (unsigned __int8 **)(v48 + 16LL * (v46 & v51));
        v50 = *v35;
        if ( v25 != *v35 )
        {
          while ( v50 != (unsigned __int8 *)-4096LL )
          {
            if ( !v44 && v50 == (unsigned __int8 *)-8192LL )
              v44 = v35;
            v49 = v46 & (v47 + v49);
            v35 = (unsigned __int8 **)(v48 + 16LL * v49);
            v50 = *v35;
            if ( v25 == *v35 )
              goto LABEL_45;
            ++v47;
          }
          goto LABEL_53;
        }
      }
      goto LABEL_45;
    }
  }
  else
  {
    ++*(_QWORD *)a3;
  }
  v53 = v27;
  sub_DA7360(a3, 2 * v28);
  v38 = *(_DWORD *)(a3 + 24);
  if ( !v38 )
    goto LABEL_74;
  v39 = v38 - 1;
  v40 = *(_QWORD *)(a3 + 8);
  v41 = v39 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
  v37 = *(_DWORD *)(a3 + 16) + 1;
  v27 = v53;
  v35 = (unsigned __int8 **)(v40 + 16LL * v41);
  v42 = *v35;
  if ( v25 != *v35 )
  {
    v43 = 1;
    v44 = 0;
    while ( v42 != (unsigned __int8 *)-4096LL )
    {
      if ( !v44 && v42 == (unsigned __int8 *)-8192LL )
        v44 = v35;
      v41 = v39 & (v43 + v41);
      v35 = (unsigned __int8 **)(v40 + 16LL * v41);
      v42 = *v35;
      if ( v25 == *v35 )
        goto LABEL_45;
      ++v43;
    }
LABEL_53:
    if ( v44 )
      v35 = v44;
  }
LABEL_45:
  *(_DWORD *)(a3 + 16) = v37;
  if ( *v35 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(a3 + 20);
  *v35 = v25;
  v32 = (__int64 *)(v35 + 1);
  v35[1] = 0;
LABEL_23:
  *v32 = v27;
  if ( v27 )
  {
    *v24 = v27;
    goto LABEL_25;
  }
  v33 = v55;
  result = 0;
LABEL_28:
  v62 = result;
  j_j___libc_free_0(v57, v33);
  return v62;
}
