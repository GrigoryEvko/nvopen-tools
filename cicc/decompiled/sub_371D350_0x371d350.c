// Function: sub_371D350
// Address: 0x371d350
//
__int64 __fastcall sub_371D350(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // r15
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // r8
  _BYTE *v7; // rdi
  __int64 v8; // r13
  unsigned int v9; // ebx
  unsigned __int8 *v10; // rax
  __int64 v11; // rsi
  int v12; // ecx
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // r9
  unsigned int v16; // esi
  __int64 v17; // rcx
  unsigned __int8 *v18; // r11
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r10
  int v22; // esi
  unsigned int v23; // r9d
  __int64 *v24; // r12
  int v25; // esi
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r11
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 *v32; // rsi
  __int64 v33; // r13
  __int64 *v34; // r15
  _DWORD *v35; // rax
  _BYTE *v36; // rdi
  unsigned int v37; // r12d
  __int64 v39; // rcx
  __int64 *v40; // r12
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 *v43; // r14
  _DWORD *v44; // rax
  __int64 v45; // rcx
  unsigned int v46; // eax
  __int64 *v47; // r14
  _DWORD *v48; // rax
  size_t v49; // rdx
  int v50; // ecx
  __int64 v51; // rax
  unsigned __int8 *v52; // rdx
  int v53; // ecx
  unsigned int v54; // r8d
  __int64 v55; // r9
  unsigned int v56; // edi
  __int64 v57; // rcx
  unsigned __int8 *v58; // rbx
  int v59; // eax
  int v60; // edx
  int v61; // edx
  int v62; // ecx
  int v63; // r12d
  __int64 v64; // [rsp+8h] [rbp-A8h]
  __int64 v67; // [rsp+28h] [rbp-88h]
  _BYTE *v68; // [rsp+30h] [rbp-80h] BYREF
  __int64 v69; // [rsp+38h] [rbp-78h]
  _BYTE dest[48]; // [rsp+40h] [rbp-70h] BYREF
  int v71; // [rsp+70h] [rbp-40h]

  v3 = a1;
  v4 = *(_QWORD *)(*a2 + 40);
  if ( v4 == *(_QWORD *)(*a3 + 40LL) )
  {
    v51 = *(_QWORD *)(*a3 + 16LL);
    if ( !v51 )
      return 0;
    while ( 1 )
    {
      v52 = *(unsigned __int8 **)(v51 + 24);
      if ( v4 != *((_QWORD *)v52 + 5) )
        return 1;
      v53 = *v52;
      if ( (_BYTE)v53 == 84 || (unsigned int)(v53 - 30) <= 0xA )
        return 1;
      v54 = *(_DWORD *)(v3 + 112);
      v55 = *(_QWORD *)(v3 + 96);
      if ( !v54 )
        goto LABEL_67;
      v56 = (v54 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      v57 = v55 + 16LL * v56;
      v58 = *(unsigned __int8 **)v57;
      if ( *(unsigned __int8 **)v57 != v52 )
        break;
LABEL_63:
      if ( *((_DWORD *)a2 + 4) < *(_DWORD *)(v57 + 8) )
        return 1;
      v51 = *(_QWORD *)(v51 + 8);
      if ( !v51 )
        return 0;
    }
    v62 = 1;
    while ( v58 != (unsigned __int8 *)-4096LL )
    {
      v63 = v62 + 1;
      v56 = (v54 - 1) & (v62 + v56);
      v57 = v55 + 16LL * v56;
      v58 = *(unsigned __int8 **)v57;
      if ( v52 == *(unsigned __int8 **)v57 )
        goto LABEL_63;
      v62 = v63;
    }
LABEL_67:
    v57 = v55 + 16LL * v54;
    goto LABEL_63;
  }
  v5 = sub_2C6ED70(*(_QWORD *)(a1 + 16), (__int64 *)a2[1]);
  v7 = dest;
  v8 = v5;
  v9 = *(_DWORD *)(v5 + 16);
  v68 = dest;
  v69 = 0x600000000LL;
  if ( v9 && &v68 != (_BYTE **)(v5 + 8) )
  {
    v49 = 8LL * v9;
    if ( v9 <= 6
      || (sub_C8D5F0((__int64)&v68, dest, v9, 8u, v6, (__int64)&v68),
          v7 = v68,
          (v49 = 8LL * *(unsigned int *)(v8 + 16)) != 0) )
    {
      memcpy(v7, *(const void **)(v8 + 8), v49);
      v7 = v68;
    }
    LODWORD(v69) = v9;
  }
  v71 = *(_DWORD *)(v8 + 72);
  *(_QWORD *)&v7[8 * (*(_DWORD *)v8 >> 6)] |= 1LL << *(_DWORD *)v8;
  v67 = *(_QWORD *)(*a3 + 16LL);
  if ( !v67 )
  {
LABEL_34:
    v36 = v68;
    v37 = 0;
LABEL_45:
    if ( v36 != dest )
      goto LABEL_23;
    return v37;
  }
  while ( 1 )
  {
    v10 = *(unsigned __int8 **)(v67 + 24);
    v11 = *a2;
    v12 = *v10;
    if ( (_BYTE)v12 == 84 )
    {
      v13 = *(_QWORD *)(*((_QWORD *)v10 - 1)
                      + 32LL * *((unsigned int *)v10 + 18)
                      + 8LL * (unsigned int)((v67 - *((_QWORD *)v10 - 1)) >> 5));
      if ( v13 == *(_QWORD *)(v11 + 40) )
      {
LABEL_68:
        v36 = v68;
        goto LABEL_22;
      }
LABEL_26:
      v39 = *(_QWORD *)(v3 + 8);
      v40 = (__int64 *)a3[1];
      if ( v13 )
      {
        v41 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
        v42 = *(_DWORD *)(v13 + 44) + 1;
      }
      else
      {
        v41 = 0;
        v42 = 0;
      }
      v43 = 0;
      if ( v42 < *(_DWORD *)(v39 + 32) )
        v43 = *(__int64 **)(*(_QWORD *)(v39 + 24) + 8 * v41);
      while ( v40 != v43 )
      {
        v44 = (_DWORD *)sub_2C6ED70(*(_QWORD *)(v3 + 16), v43);
        v36 = v68;
        if ( (*(_QWORD *)&v68[8 * (*v44 >> 6)] & (1LL << *v44)) != 0 )
          goto LABEL_22;
        v43 = (__int64 *)v43[1];
      }
      goto LABEL_33;
    }
    v13 = *((_QWORD *)v10 + 5);
    if ( v13 != *(_QWORD *)(v11 + 40) )
      goto LABEL_26;
    if ( (unsigned int)(v12 - 30) <= 0xA )
      goto LABEL_68;
    v14 = *(unsigned int *)(v3 + 112);
    v15 = *(_QWORD *)(v3 + 96);
    if ( (_DWORD)v14 )
    {
      v16 = (v14 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v17 = v15 + 16LL * v16;
      v18 = *(unsigned __int8 **)v17;
      if ( v10 == *(unsigned __int8 **)v17 )
        goto LABEL_9;
      v50 = 1;
      while ( v18 != (unsigned __int8 *)-4096LL )
      {
        v61 = v50 + 1;
        v16 = (v14 - 1) & (v50 + v16);
        v17 = v15 + 16LL * v16;
        v18 = *(unsigned __int8 **)v17;
        if ( v10 == *(unsigned __int8 **)v17 )
          goto LABEL_9;
        v50 = v61;
      }
    }
    v17 = v15 + 16 * v14;
LABEL_9:
    if ( *((_DWORD *)a2 + 4) < *(_DWORD *)(v17 + 8) )
      goto LABEL_68;
    v19 = *(_QWORD *)(v3 + 24);
    v20 = *(_QWORD *)(v3 + 8);
    v21 = *(_QWORD *)(v19 + 8);
    v22 = *(_DWORD *)(v19 + 24);
    v23 = *(_DWORD *)(v20 + 32);
    v24 = (__int64 *)a3[1];
    if ( !v22 )
      goto LABEL_36;
    v25 = v22 - 1;
    v26 = v25 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v27 = (__int64 *)(v21 + 16LL * v26);
    v28 = *v27;
    if ( v13 == *v27 )
      break;
    v59 = 1;
    while ( v28 != -4096 )
    {
      v60 = v59 + 1;
      v26 = v25 & (v59 + v26);
      v27 = (__int64 *)(v21 + 16LL * v26);
      v28 = *v27;
      if ( v13 == *v27 )
        goto LABEL_12;
      v59 = v60;
    }
LABEL_36:
    if ( v13 )
    {
      v45 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
      v46 = *(_DWORD *)(v13 + 44) + 1;
    }
    else
    {
      v45 = 0;
      v46 = 0;
    }
    v47 = 0;
    if ( v23 > v46 )
      v47 = *(__int64 **)(*(_QWORD *)(v20 + 24) + 8 * v45);
    if ( v24 != v47 )
    {
      while ( 1 )
      {
        v48 = (_DWORD *)sub_2C6ED70(*(_QWORD *)(v3 + 16), v47);
        if ( (*(_QWORD *)(*(_QWORD *)(v8 + 8) + 8LL * (*v48 >> 6)) & (1LL << *v48)) != 0 )
          break;
        v47 = (__int64 *)v47[1];
        if ( v24 == v47 )
          goto LABEL_33;
      }
      v36 = v68;
      v37 = 1;
      goto LABEL_45;
    }
LABEL_33:
    v67 = *(_QWORD *)(v67 + 8);
    if ( !v67 )
      goto LABEL_34;
  }
LABEL_12:
  v29 = v27[1];
  if ( !v29 || v13 != **(_QWORD **)(v29 + 32) )
    goto LABEL_36;
  if ( v13 )
  {
    v30 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
    v31 = *(_DWORD *)(v13 + 44) + 1;
  }
  else
  {
    v30 = 0;
    v31 = 0;
  }
  v32 = 0;
  if ( v23 > v31 )
    v32 = *(__int64 **)(*(_QWORD *)(v20 + 24) + 8 * v30);
  if ( v24 == v32 )
    goto LABEL_36;
  v64 = v8;
  v33 = v3;
  v34 = v32;
  while ( 1 )
  {
    v35 = (_DWORD *)sub_2C6ED70(*(_QWORD *)(v33 + 16), v34);
    v36 = v68;
    if ( (*(_QWORD *)&v68[8 * (*v35 >> 6)] & (1LL << *v35)) != 0 )
      break;
    v34 = (__int64 *)v34[1];
    if ( v24 == v34 )
    {
      v3 = v33;
      v8 = v64;
      v20 = *(_QWORD *)(v3 + 8);
      v24 = (__int64 *)a3[1];
      v23 = *(_DWORD *)(v20 + 32);
      goto LABEL_36;
    }
  }
LABEL_22:
  v37 = 1;
  if ( v36 == dest )
    return v37;
LABEL_23:
  _libc_free((unsigned __int64)v36);
  return v37;
}
