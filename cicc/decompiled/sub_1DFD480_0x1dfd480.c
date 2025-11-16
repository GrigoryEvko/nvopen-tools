// Function: sub_1DFD480
// Address: 0x1dfd480
//
unsigned __int64 __fastcall sub_1DFD480(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r14
  void *v9; // r8
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // r13
  __int64 *v14; // rbx
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned int v21; // edi
  __int64 *v22; // rsi
  __int64 v23; // r10
  unsigned int v24; // r15d
  __int64 v25; // r8
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r8
  unsigned int v33; // edi
  __int64 *v34; // rcx
  __int64 v35; // r10
  int v36; // edi
  int v37; // edi
  __int64 v38; // r8
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r10
  __int64 *v42; // rax
  __int64 v43; // rdx
  int v44; // ecx
  int v45; // r8d
  int v46; // r9d
  unsigned __int64 v47; // rsi
  unsigned int v48; // ecx
  unsigned __int64 v49; // rbx
  unsigned __int64 result; // rax
  __int64 v51; // r14
  unsigned __int64 *v52; // rax
  unsigned __int64 *v53; // rdx
  __int64 v54; // rbx
  unsigned __int64 v55; // r15
  void *v56; // r14
  __int64 v57; // rax
  void *v58; // rdi
  int v59; // eax
  int v60; // r9d
  int v61; // esi
  int v62; // ecx
  int v63; // eax
  int v64; // r11d
  int v65; // r11d
  int v66; // r9d
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // [rsp+0h] [rbp-60h]
  __int64 v70; // [rsp+8h] [rbp-58h]
  __int64 v71; // [rsp+8h] [rbp-58h]
  void *v72; // [rsp+8h] [rbp-58h]
  unsigned int v73; // [rsp+10h] [rbp-50h]
  __int64 *v74; // [rsp+10h] [rbp-50h]
  __int64 v75[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v76[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88);
  v75[0] = a2;
  v4 = v3 >> 2;
  v5 = sub_22077B0(72);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    v7 = (unsigned int)(v4 + 63) >> 6;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_DWORD *)(v5 + 40) = v4;
    v73 = (unsigned int)(v4 + 63) >> 6;
    v8 = 8 * v7;
    v9 = (void *)malloc(8 * v7);
    if ( !v9 )
    {
      if ( v8 || (v68 = malloc(1u), v9 = 0, !v68) )
      {
        v72 = v9;
        sub_16BD1C0("Allocation failed", 1u);
        v9 = v72;
      }
      else
      {
        v9 = (void *)v68;
      }
    }
    *(_QWORD *)(v6 + 24) = v9;
    *(_QWORD *)(v6 + 32) = v7;
    if ( v73 )
    {
      memset(v9, 0, 8 * v7);
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 56) = 0;
      *(_DWORD *)(v6 + 64) = v4;
      v57 = malloc(8 * v7);
      v58 = (void *)v57;
      if ( v57 )
      {
        *(_QWORD *)(v6 + 48) = v57;
        *(_QWORD *)(v6 + 56) = v7;
LABEL_48:
        memset(v58, 0, 8 * v7);
        goto LABEL_6;
      }
    }
    else
    {
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 56) = 0;
      *(_DWORD *)(v6 + 64) = v4;
      v10 = malloc(8 * v7);
      if ( v10 )
      {
        *(_QWORD *)(v6 + 48) = v10;
        *(_QWORD *)(v6 + 56) = 0;
        goto LABEL_6;
      }
    }
    if ( v8 || (v58 = (void *)malloc(1u)) == 0 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v58 = 0;
    }
    *(_QWORD *)(v6 + 48) = v58;
    *(_QWORD *)(v6 + 56) = v7;
    if ( v73 )
      goto LABEL_48;
  }
LABEL_6:
  v69 = a1 + 112;
  v11 = sub_1DFD350(a1 + 112, v75);
  v12 = v11[1];
  v11[1] = v6;
  if ( v12 )
  {
    _libc_free(*(_QWORD *)(v12 + 48));
    _libc_free(*(_QWORD *)(v12 + 24));
    j_j___libc_free_0(v12, 72);
  }
  v13 = sub_1DFD350(v69, v75)[1];
  v14 = *(__int64 **)(v75[0] + 88);
  v74 = *(__int64 **)(v75[0] + 96);
  if ( v14 != v74 )
  {
    while ( 1 )
    {
      v76[0] = *v14;
      sub_21EBA40(a1);
      v15 = v76[0];
      v16 = v75[0];
      if ( v76[0] != v75[0] )
        break;
LABEL_25:
      if ( v74 == ++v14 )
        goto LABEL_26;
    }
    v70 = *(_QWORD *)(a1 + 16);
    sub_1E06620(v70);
    v17 = 0;
    v18 = *(_QWORD *)(v70 + 1312);
    v19 = *(unsigned int *)(v18 + 48);
    if ( (_DWORD)v19 )
    {
      v20 = *(_QWORD *)(v18 + 32);
      v21 = (v19 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v16 == *v22 )
      {
LABEL_12:
        if ( v22 != (__int64 *)(v20 + 16 * v19) )
        {
          v17 = v22[1];
          goto LABEL_14;
        }
      }
      else
      {
        v61 = 1;
        while ( v23 != -8 )
        {
          v65 = v61 + 1;
          v21 = (v19 - 1) & (v61 + v21);
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v16 == *v22 )
            goto LABEL_12;
          v61 = v65;
        }
      }
      v17 = 0;
    }
LABEL_14:
    v24 = *(_DWORD *)(a1 + 168);
    if ( v24 )
    {
      v25 = *(_QWORD *)(a1 + 152);
      v26 = (v24 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v17 == *v27 )
      {
LABEL_16:
        v24 = *((_DWORD *)v27 + 2);
      }
      else
      {
        v63 = 1;
        while ( v28 != -8 )
        {
          v66 = v63 + 1;
          v26 = (v24 - 1) & (v63 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v17 == *v27 )
            goto LABEL_16;
          v63 = v66;
        }
        v24 = 0;
      }
    }
    v71 = *(_QWORD *)(a1 + 16);
    sub_1E06620(v71);
    v29 = 0;
    v30 = *(_QWORD *)(v71 + 1312);
    v31 = *(unsigned int *)(v30 + 48);
    if ( !(_DWORD)v31 )
      goto LABEL_21;
    v32 = *(_QWORD *)(v30 + 32);
    v33 = (v31 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v34 = (__int64 *)(v32 + 16LL * v33);
    v35 = *v34;
    if ( v15 == *v34 )
    {
LABEL_19:
      if ( v34 != (__int64 *)(v32 + 16 * v31) )
      {
        v29 = v34[1];
        goto LABEL_21;
      }
    }
    else
    {
      v62 = 1;
      while ( v35 != -8 )
      {
        v64 = v62 + 1;
        v33 = (v31 - 1) & (v62 + v33);
        v34 = (__int64 *)(v32 + 16LL * v33);
        v35 = *v34;
        if ( v15 == *v34 )
          goto LABEL_19;
        v62 = v64;
      }
    }
    v29 = 0;
LABEL_21:
    v36 = *(_DWORD *)(a1 + 168);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 152);
      v39 = v37 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v40 = (__int64 *)(v38 + 16LL * v39);
      v41 = *v40;
      if ( *v40 == v29 )
      {
LABEL_23:
        if ( *((_DWORD *)v40 + 2) > v24 )
        {
          v42 = sub_1DFD350(v69, v76);
          sub_1DF8070(v13 + 48, v42[1] + 24, v43, v44, v45, v46);
          sub_21EAF50(a1, v75[0], v76[0]);
        }
      }
      else
      {
        v59 = 1;
        while ( v41 != -8 )
        {
          v60 = v59 + 1;
          v39 = v37 & (v59 + v39);
          v40 = (__int64 *)(v38 + 16LL * v39);
          v41 = *v40;
          if ( v29 == *v40 )
            goto LABEL_23;
          v59 = v60;
        }
      }
    }
    goto LABEL_25;
  }
LABEL_26:
  v47 = *(unsigned int *)(v13 + 64);
  *(_DWORD *)(v13 + 40) = v47;
  v48 = (unsigned int)(v47 + 63) >> 6;
  if ( v47 > *(_QWORD *)(v13 + 32) << 6 )
  {
    v54 = v48;
    v55 = 8LL * v48;
    v56 = (void *)malloc(v55);
    if ( !v56 )
    {
      if ( v55 || (v67 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v56 = (void *)v67;
    }
    memcpy(v56, *(const void **)(v13 + 48), v55);
    _libc_free(*(_QWORD *)(v13 + 24));
    *(_QWORD *)(v13 + 24) = v56;
    *(_QWORD *)(v13 + 32) = v54;
  }
  else
  {
    if ( (_DWORD)v47 )
      memcpy(*(void **)(v13 + 24), *(const void **)(v13 + 48), 8LL * v48);
    sub_13A4C60(v13 + 24, 0);
  }
  v49 = *(_QWORD *)(v75[0] + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v49 )
LABEL_82:
    BUG();
  result = *(_QWORD *)v49;
  if ( (*(_QWORD *)v49 & 4) == 0 && (*(_BYTE *)(v49 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v49 = result;
      if ( (*(_BYTE *)(result + 46) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
  }
  v51 = v75[0] + 24;
  while ( v51 != v49 )
  {
    sub_21ECAD0(a1, v49, v13);
    v52 = (unsigned __int64 *)(*(_QWORD *)v49 & 0xFFFFFFFFFFFFFFF8LL);
    v53 = v52;
    if ( !v52 )
      goto LABEL_82;
    v49 = *(_QWORD *)v49 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v52;
    if ( (result & 4) == 0 && (*((_BYTE *)v53 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v49 = result;
        if ( (*(_BYTE *)(result + 46) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
