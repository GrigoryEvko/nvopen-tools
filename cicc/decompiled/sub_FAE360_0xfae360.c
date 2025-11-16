// Function: sub_FAE360
// Address: 0xfae360
//
__int64 __fastcall sub_FAE360(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned int v10; // esi
  __int64 v11; // r9
  int v12; // r11d
  unsigned int v13; // edi
  __int64 *v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r12
  char v18; // cl
  __int64 v19; // r10
  int v20; // esi
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // r11
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r12
  int v27; // eax
  __int64 *v28; // rdx
  __int64 v29; // rax
  unsigned int v30; // ebx
  int v32; // eax
  int v33; // ecx
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  unsigned int v36; // esi
  unsigned int v37; // eax
  __int64 *v38; // r9
  int v39; // edx
  unsigned int v40; // edi
  __int64 v41; // rdx
  int v42; // r14d
  int v43; // ecx
  __int64 v44; // rsi
  int v45; // ecx
  __int64 v46; // rax
  __int64 v47; // rdi
  int v48; // eax
  int v49; // esi
  __int64 v50; // rdi
  unsigned int v51; // eax
  __int64 v52; // r9
  int v53; // r11d
  __int64 *v54; // r10
  int v55; // eax
  int v56; // esi
  __int64 v57; // rdi
  __int64 *v58; // r9
  unsigned int v59; // r14d
  int v60; // r10d
  __int64 v61; // rax
  int v62; // ecx
  __int64 v63; // rsi
  int v64; // ecx
  __int64 v65; // rax
  __int64 v66; // rdi
  int v67; // r11d
  __int64 *v68; // r10
  int v69; // r11d
  __int64 v70; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v71; // [rsp-80h] [rbp-80h] BYREF
  __int64 *v72; // [rsp-78h] [rbp-78h] BYREF
  __int64 v73; // [rsp-70h] [rbp-70h]
  _BYTE v74[104]; // [rsp-68h] [rbp-68h] BYREF

  v1 = *(_QWORD *)(*a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == *a1 + 48LL )
    goto LABEL_114;
  if ( !v1 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 > 0xA )
LABEL_114:
    BUG();
  v3 = *(_QWORD *)(v1 - 56);
  v72 = (__int64 *)v74;
  v70 = v3;
  v73 = 0x600000000LL;
  v4 = sub_AA5930(v3);
  v7 = v6;
  v8 = v4;
LABEL_5:
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      v9 = a1[1];
      v10 = *(_DWORD *)(v9 + 24);
      if ( !v10 )
        break;
      v11 = *(_QWORD *)(v9 + 8);
      v12 = 1;
      v13 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v14 = 0;
      v15 = (__int64 *)(v11 + 152LL * v13);
      v16 = *v15;
      if ( *v15 == v8 )
      {
LABEL_8:
        v17 = (__int64)(v15 + 1);
        v18 = v15[2] & 1;
        if ( v18 )
          goto LABEL_9;
        goto LABEL_39;
      }
      while ( v16 != -4096 )
      {
        if ( !v14 && v16 == -8192 )
          v14 = v15;
        v5 = (unsigned int)(v12 + 1);
        v13 = (v10 - 1) & (v12 + v13);
        v15 = (__int64 *)(v11 + 152LL * v13);
        v16 = *v15;
        if ( *v15 == v8 )
          goto LABEL_8;
        ++v12;
      }
      if ( !v14 )
        v14 = v15;
      v32 = *(_DWORD *)(v9 + 16);
      ++*(_QWORD *)v9;
      v33 = v32 + 1;
      if ( 4 * (v32 + 1) >= 3 * v10 )
        goto LABEL_64;
      if ( v10 - *(_DWORD *)(v9 + 20) - v33 <= v10 >> 3 )
      {
        sub_F9EFC0(v9, v10);
        v55 = *(_DWORD *)(v9 + 24);
        if ( !v55 )
        {
LABEL_117:
          ++*(_DWORD *)(v9 + 16);
          BUG();
        }
        v56 = v55 - 1;
        v57 = *(_QWORD *)(v9 + 8);
        v58 = 0;
        v59 = (v55 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v60 = 1;
        v33 = *(_DWORD *)(v9 + 16) + 1;
        v14 = (__int64 *)(v57 + 152LL * v59);
        v61 = *v14;
        if ( *v14 != v8 )
        {
          while ( v61 != -4096 )
          {
            if ( v61 == -8192 && !v58 )
              v58 = v14;
            v59 = v56 & (v60 + v59);
            v14 = (__int64 *)(v57 + 152LL * v59);
            v61 = *v14;
            if ( *v14 == v8 )
              goto LABEL_32;
            ++v60;
          }
          if ( v58 )
            v14 = v58;
        }
      }
LABEL_32:
      *(_DWORD *)(v9 + 16) = v33;
      if ( *v14 != -4096 )
        --*(_DWORD *)(v9 + 20);
      *v14 = v8;
      v17 = (__int64)(v14 + 1);
      v34 = v14 + 3;
      v35 = v14 + 19;
      *(v35 - 18) = 0;
      *(v35 - 17) = 1;
      do
      {
        if ( v34 )
          *v34 = -4096;
        v34 += 2;
      }
      while ( v34 != v35 );
      v18 = *(_BYTE *)(v17 + 8) & 1;
      if ( v18 )
      {
LABEL_9:
        v19 = v17 + 16;
        v20 = 7;
        goto LABEL_10;
      }
LABEL_39:
      v36 = *(_DWORD *)(v17 + 24);
      v19 = *(_QWORD *)(v17 + 16);
      if ( !v36 )
      {
        v37 = *(_DWORD *)(v17 + 8);
        ++*(_QWORD *)v17;
        v38 = 0;
        v39 = (v37 >> 1) + 1;
LABEL_42:
        v40 = 3 * v36;
        goto LABEL_43;
      }
      v20 = v36 - 1;
LABEL_10:
      v21 = v20 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v22 = (__int64 *)(v19 + 16 * v21);
      v23 = *v22;
      if ( *v22 == v70 )
      {
LABEL_11:
        v24 = (unsigned int)v73;
        v25 = HIDWORD(v73);
        v26 = v22[1];
        v27 = v73;
        if ( (unsigned int)v73 < (unsigned __int64)HIDWORD(v73) )
          goto LABEL_12;
        goto LABEL_49;
      }
      v42 = 1;
      v38 = 0;
      while ( v23 != -4096 )
      {
        if ( !v38 && v23 == -8192 )
          v38 = v22;
        v5 = (unsigned int)(v42 + 1);
        LODWORD(v21) = v20 & (v42 + v21);
        v22 = (__int64 *)(v19 + 16LL * (unsigned int)v21);
        v23 = *v22;
        if ( v70 == *v22 )
          goto LABEL_11;
        ++v42;
      }
      v40 = 24;
      v36 = 8;
      if ( !v38 )
        v38 = v22;
      v37 = *(_DWORD *)(v17 + 8);
      ++*(_QWORD *)v17;
      v39 = (v37 >> 1) + 1;
      if ( !v18 )
      {
        v36 = *(_DWORD *)(v17 + 24);
        goto LABEL_42;
      }
LABEL_43:
      if ( 4 * v39 >= v40 )
      {
        sub_FADF20(v17, 2 * v36);
        if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
        {
          v44 = v17 + 16;
          v45 = 7;
        }
        else
        {
          v43 = *(_DWORD *)(v17 + 24);
          v44 = *(_QWORD *)(v17 + 16);
          if ( !v43 )
            goto LABEL_118;
          v45 = v43 - 1;
        }
        v41 = v70;
        v46 = v45 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
        v38 = (__int64 *)(v44 + 16 * v46);
        v47 = *v38;
        if ( *v38 == v70 )
          goto LABEL_62;
        v69 = 1;
        v68 = 0;
        while ( v47 != -4096 )
        {
          if ( !v68 && v47 == -8192 )
            v68 = v38;
          LODWORD(v46) = v45 & (v69 + v46);
          v38 = (__int64 *)(v44 + 16LL * (unsigned int)v46);
          v47 = *v38;
          if ( v70 == *v38 )
            goto LABEL_62;
          ++v69;
        }
        goto LABEL_83;
      }
      if ( v36 - *(_DWORD *)(v17 + 12) - v39 > v36 >> 3 )
      {
        v41 = v70;
        goto LABEL_46;
      }
      sub_FADF20(v17, v36);
      if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
      {
        v63 = v17 + 16;
        v64 = 7;
      }
      else
      {
        v62 = *(_DWORD *)(v17 + 24);
        v63 = *(_QWORD *)(v17 + 16);
        if ( !v62 )
        {
LABEL_118:
          *(_DWORD *)(v17 + 8) = (2 * (*(_DWORD *)(v17 + 8) >> 1) + 2) | *(_DWORD *)(v17 + 8) & 1;
          BUG();
        }
        v64 = v62 - 1;
      }
      v41 = v70;
      v65 = v64 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v38 = (__int64 *)(v63 + 16 * v65);
      v66 = *v38;
      if ( *v38 != v70 )
      {
        v67 = 1;
        v68 = 0;
        while ( v66 != -4096 )
        {
          if ( !v68 && v66 == -8192 )
            v68 = v38;
          LODWORD(v65) = v64 & (v67 + v65);
          v38 = (__int64 *)(v63 + 16LL * (unsigned int)v65);
          v66 = *v38;
          if ( v70 == *v38 )
            goto LABEL_62;
          ++v67;
        }
LABEL_83:
        if ( v68 )
          v38 = v68;
      }
LABEL_62:
      v37 = *(_DWORD *)(v17 + 8);
LABEL_46:
      *(_DWORD *)(v17 + 8) = (2 * (v37 >> 1) + 2) | v37 & 1;
      if ( *v38 != -4096 )
        --*(_DWORD *)(v17 + 12);
      *v38 = v41;
      v26 = 0;
      v38[1] = 0;
      v24 = (unsigned int)v73;
      v25 = HIDWORD(v73);
      v27 = v73;
      if ( (unsigned int)v73 < (unsigned __int64)HIDWORD(v73) )
      {
LABEL_12:
        v28 = &v72[v24];
        if ( v28 )
        {
          *v28 = v26;
          v27 = v73;
        }
        LODWORD(v73) = v27 + 1;
        goto LABEL_15;
      }
LABEL_49:
      if ( v25 < v24 + 1 )
      {
        sub_C8D5F0((__int64)&v72, v74, v24 + 1, 8u, v5, v24 + 1);
        v24 = (unsigned int)v73;
      }
      v72[v24] = v26;
      LODWORD(v73) = v73 + 1;
LABEL_15:
      if ( !v8 )
        BUG();
      v29 = *(_QWORD *)(v8 + 32);
      if ( !v29 )
        BUG();
      v8 = 0;
      if ( *(_BYTE *)(v29 - 24) != 84 )
        goto LABEL_5;
      v8 = v29 - 24;
      if ( v7 == v29 - 24 )
        goto LABEL_19;
    }
    ++*(_QWORD *)v9;
LABEL_64:
    sub_F9EFC0(v9, 2 * v10);
    v48 = *(_DWORD *)(v9 + 24);
    if ( !v48 )
      goto LABEL_117;
    v49 = v48 - 1;
    v50 = *(_QWORD *)(v9 + 8);
    v51 = (v48 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v14 = (__int64 *)(v50 + 152LL * v51);
    v52 = *v14;
    v33 = *(_DWORD *)(v9 + 16) + 1;
    if ( *v14 != v8 )
    {
      v53 = 1;
      v54 = 0;
      while ( v52 != -4096 )
      {
        if ( !v54 && v52 == -8192 )
          v54 = v14;
        v51 = v49 & (v53 + v51);
        v14 = (__int64 *)(v50 + 152LL * v51);
        v52 = *v14;
        if ( *v14 == v8 )
          goto LABEL_32;
        ++v53;
      }
      if ( v54 )
        v14 = v54;
    }
    goto LABEL_32;
  }
LABEL_19:
  v71 = sub_F9DD90(v72, (__int64)&v72[(unsigned int)v73]);
  v30 = sub_F9AB30(&v70, (__int64 *)&v71);
  if ( v72 != (__int64 *)v74 )
    _libc_free(v72, &v71);
  return v30;
}
