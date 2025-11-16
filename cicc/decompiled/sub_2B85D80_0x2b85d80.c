// Function: sub_2B85D80
// Address: 0x2b85d80
//
_QWORD *__fastcall sub_2B85D80(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // eax
  int v9; // ecx
  unsigned int v10; // eax
  __int64 *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // r8
  __int64 v20; // r9
  char v21; // si
  __int64 v22; // rdi
  int v23; // eax
  unsigned int v24; // edx
  __int64 v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rax
  __int64 *v28; // rbx
  __int64 *v29; // r15
  __int64 *v30; // rax
  __int64 v31; // r12
  __int64 v32; // rdi
  int v33; // eax
  int v34; // edx
  __int64 v35; // rdx
  bool v36; // al
  __int64 v37; // rdx
  __int64 i; // rdx
  _QWORD *v39; // rax
  __int64 v41; // r8
  __int64 *v42; // rbx
  __int64 *v43; // r9
  __int64 *v44; // r10
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // edx
  int v49; // eax
  __int64 v50; // rdx
  bool v51; // al
  __int64 v52; // rdx
  __int64 v53; // rdx
  _QWORD *v54; // rdi
  unsigned int v55; // eax
  int v56; // r12d
  unsigned int v57; // eax
  _QWORD *v58; // rax
  int v59; // edi
  bool v60; // al
  bool v61; // al
  __int64 v62; // rcx
  int v63; // ecx
  _QWORD *v64; // rax
  int v65; // r11d
  __int64 *v66; // [rsp+0h] [rbp-F0h]
  __int64 *v67; // [rsp+0h] [rbp-F0h]
  __int64 *v68; // [rsp+8h] [rbp-E8h]
  __int64 *v69; // [rsp+8h] [rbp-E8h]
  __int64 *v70; // [rsp+8h] [rbp-E8h]
  __int64 v71; // [rsp+10h] [rbp-E0h]
  __int64 v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 *v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+10h] [rbp-E0h]
  __int64 v76; // [rsp+10h] [rbp-E0h]
  __int64 v77; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD *v78; // [rsp+28h] [rbp-C8h]
  __int64 v79; // [rsp+30h] [rbp-C0h]
  __int64 v80; // [rsp+38h] [rbp-B8h]
  char *v81; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v82; // [rsp+48h] [rbp-A8h]
  _BYTE v83[32]; // [rsp+50h] [rbp-A0h] BYREF
  __int128 v84; // [rsp+70h] [rbp-80h]
  __int128 v85; // [rsp+80h] [rbp-70h]
  __int64 *v86; // [rsp+90h] [rbp-60h] BYREF
  __int64 v87; // [rsp+98h] [rbp-58h]
  _BYTE v88[80]; // [rsp+A0h] [rbp-50h] BYREF

  v81 = v83;
  v82 = 0x400000000LL;
  v6 = *a2;
  v77 = 0;
  v7 = *(_QWORD *)(v6 + 1176);
  v8 = *(_DWORD *)(v6 + 1192);
  v78 = 0;
  v79 = 0;
  v80 = 0;
  if ( !v8 )
  {
LABEL_74:
    v84 = 0;
    v85 = 0;
    goto LABEL_6;
  }
  v9 = v8 - 1;
  v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v11 = (__int64 *)(v7 + 88LL * v10);
  v12 = *v11;
  if ( a3 != *v11 )
  {
    v59 = 1;
    while ( v12 != -4096 )
    {
      v10 = v9 & (v59 + v10);
      v11 = (__int64 *)(v7 + 88LL * v10);
      v12 = *v11;
      if ( a3 == *v11 )
        goto LABEL_3;
      ++v59;
    }
    goto LABEL_74;
  }
LABEL_3:
  v84 = 0u;
  v85 = 0u;
  sub_C7D6A0(0, 0, 8);
  v17 = *((unsigned int *)v11 + 8);
  DWORD2(v85) = v17;
  if ( (_DWORD)v17 )
  {
    *((_QWORD *)&v84 + 1) = sub_C7D670(8 * v17, 8);
    *(_QWORD *)&v85 = v11[3];
    memcpy(*((void **)&v84 + 1), (const void *)v11[2], 8LL * DWORD2(v85));
  }
  else
  {
    *((_QWORD *)&v84 + 1) = 0;
    *(_QWORD *)&v85 = 0;
  }
  v87 = 0x400000000LL;
  v18 = *((_DWORD *)v11 + 12);
  v86 = (__int64 *)v88;
  if ( !v18 )
    goto LABEL_6;
  sub_2B0C210((__int64)&v86, (__int64)(v11 + 5), v13, v14, v15, v16);
  v42 = v86;
  v43 = &v86[(unsigned int)v87];
  if ( v86 != v43 )
  {
    v44 = &v77;
    while ( 1 )
    {
      while ( 1 )
      {
        v45 = (__int64 *)a2[1];
        v46 = *v42;
        v47 = *v45;
        if ( *v45 == *v42 || !*(_QWORD *)(v46 + 184) )
          goto LABEL_40;
        v48 = *(_DWORD *)(v46 + 120);
        if ( !v48 )
          v48 = *(_DWORD *)(v46 + 8);
        v49 = *(_DWORD *)(v47 + 120);
        if ( !v49 )
          v49 = *(_DWORD *)(v47 + 8);
        if ( v48 != v49 )
          goto LABEL_40;
        v50 = *(unsigned int *)(v46 + 8);
        if ( *(_DWORD *)(v47 + 8) != (_DWORD)v50 )
          goto LABEL_40;
        if ( !*(_DWORD *)(v46 + 152) )
          break;
LABEL_50:
        if ( !*(_DWORD *)(v47 + 152) )
        {
          v66 = v44;
          v68 = v43;
          v73 = v46;
          v51 = sub_2B31C30(v46, *(char **)v47, *(unsigned int *)(v47 + 8), v46, v41, (__int64)v43);
          v46 = v73;
          v43 = v68;
          v44 = v66;
          if ( v51 )
            goto LABEL_52;
        }
LABEL_40:
        if ( v43 == ++v42 )
          goto LABEL_53;
      }
      v67 = v44;
      v70 = v43;
      v76 = *v42;
      v61 = sub_2B31C30(v47, *(char **)v46, v50, v46, v41, (__int64)v43);
      v46 = v76;
      v43 = v70;
      v44 = v67;
      if ( !v61 )
      {
        v47 = *(_QWORD *)a2[1];
        goto LABEL_50;
      }
LABEL_52:
      ++v42;
      v69 = v43;
      v74 = v44;
      sub_2B85740((__int64)v44, (__int64 *)(v46 + 184), v52, v46, v41, (__int64)v43);
      v43 = v69;
      v44 = v74;
      if ( v69 == v42 )
      {
LABEL_53:
        v43 = v86;
        break;
      }
    }
  }
  if ( v43 != (__int64 *)v88 )
    _libc_free((unsigned __int64)v43);
LABEL_6:
  sub_C7D6A0(*((__int64 *)&v84 + 1), 8LL * DWORD2(v85), 8);
  v20 = *a2;
  v21 = *(_BYTE *)(*a2 + 88) & 1;
  if ( v21 )
  {
    v22 = v20 + 96;
    v23 = 3;
  }
  else
  {
    v53 = *(unsigned int *)(v20 + 104);
    v22 = *(_QWORD *)(v20 + 96);
    if ( !(_DWORD)v53 )
      goto LABEL_80;
    v23 = v53 - 1;
  }
  v24 = v23 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v25 = v22 + 72LL * v24;
  v26 = *(_QWORD *)v25;
  if ( a3 == *(_QWORD *)v25 )
    goto LABEL_9;
  v63 = 1;
  while ( v26 != -4096 )
  {
    v65 = v63 + 1;
    v24 = v23 & (v63 + v24);
    v25 = v22 + 72LL * v24;
    v26 = *(_QWORD *)v25;
    if ( a3 == *(_QWORD *)v25 )
      goto LABEL_9;
    v63 = v65;
  }
  if ( v21 )
  {
    v62 = 288;
    goto LABEL_81;
  }
  v53 = *(unsigned int *)(v20 + 104);
LABEL_80:
  v62 = 72 * v53;
LABEL_81:
  v25 = v22 + v62;
LABEL_9:
  v27 = 288;
  if ( !v21 )
    v27 = 72LL * *(unsigned int *)(v20 + 104);
  if ( v25 != v22 + v27 )
  {
    v28 = *(__int64 **)(v25 + 8);
    v29 = &v28[*(unsigned int *)(v25 + 16)];
    if ( v29 != v28 )
    {
      v25 = (__int64)&v77;
      do
      {
        while ( 1 )
        {
          v30 = (__int64 *)a2[1];
          v31 = *v28;
          v32 = *v30;
          if ( *v30 == *v28 || !*(_QWORD *)(v31 + 184) )
            goto LABEL_14;
          v33 = *(_DWORD *)(v31 + 120);
          if ( !v33 )
            v33 = *(_DWORD *)(v31 + 8);
          v34 = *(_DWORD *)(v32 + 120);
          if ( !v34 )
            v34 = *(_DWORD *)(v32 + 8);
          if ( v34 != v33 )
            goto LABEL_14;
          v35 = *(unsigned int *)(v31 + 8);
          if ( *(_DWORD *)(v32 + 8) != (_DWORD)v35 )
            goto LABEL_14;
          v19 = *(unsigned int *)(v31 + 152);
          if ( !(_DWORD)v19 )
            break;
LABEL_24:
          if ( !*(_DWORD *)(v32 + 152) )
          {
            v71 = v25;
            v36 = sub_2B31C30(v31, *(char **)v32, *(unsigned int *)(v32 + 8), v25, v19, v20);
            v25 = v71;
            if ( v36 )
              goto LABEL_26;
          }
LABEL_14:
          if ( v29 == ++v28 )
            goto LABEL_27;
        }
        v75 = v25;
        v60 = sub_2B31C30(v32, *(char **)v31, v35, v25, v19, v20);
        v25 = v75;
        if ( !v60 )
        {
          v32 = *(_QWORD *)a2[1];
          goto LABEL_24;
        }
LABEL_26:
        ++v28;
        v72 = v25;
        sub_2B85740(v25, (__int64 *)(v31 + 184), v37, v25, v19, v20);
        v25 = v72;
      }
      while ( v29 != v28 );
    }
  }
LABEL_27:
  ++v77;
  if ( !(_DWORD)v79 )
  {
    i = HIDWORD(v79);
    if ( !HIDWORD(v79) )
      goto LABEL_33;
    i = (unsigned int)v80;
    if ( (unsigned int)v80 <= 0x40 )
      goto LABEL_30;
    sub_C7D6A0((__int64)v78, 8LL * (unsigned int)v80, 8);
    LODWORD(v80) = 0;
LABEL_87:
    v78 = 0;
LABEL_32:
    v79 = 0;
    goto LABEL_33;
  }
  v25 = (unsigned int)(4 * v79);
  i = (unsigned int)v80;
  if ( (unsigned int)v25 < 0x40 )
    v25 = 64;
  if ( (unsigned int)v25 >= (unsigned int)v80 )
  {
LABEL_30:
    v39 = v78;
    i = (__int64)&v78[i];
    if ( v78 != (_QWORD *)i )
    {
      do
        *v39++ = -4096;
      while ( (_QWORD *)i != v39 );
    }
    goto LABEL_32;
  }
  v54 = v78;
  v19 = 8LL * (unsigned int)v80;
  if ( (_DWORD)v79 == 1 )
  {
    v56 = 64;
  }
  else
  {
    _BitScanReverse(&v55, v79 - 1);
    v55 ^= 0x1Fu;
    v25 = 33 - v55;
    v56 = 1 << (33 - v55);
    if ( v56 < 64 )
      v56 = 64;
    if ( v56 == (_DWORD)v80 )
    {
      v79 = 0;
      v64 = (_QWORD *)((char *)v78 + v19);
      do
      {
        if ( v54 )
          *v54 = -4096;
        ++v54;
      }
      while ( v64 != v54 );
      goto LABEL_33;
    }
  }
  sub_C7D6A0((__int64)v78, v19, 8);
  v57 = sub_2B149A0(v56);
  LODWORD(v80) = v57;
  if ( !v57 )
    goto LABEL_87;
  v58 = (_QWORD *)sub_C7D670(8LL * v57, 8);
  v79 = 0;
  v78 = v58;
  for ( i = (__int64)&v58[(unsigned int)v80]; (_QWORD *)i != v58; ++v58 )
  {
    if ( v58 )
      *v58 = -4096;
  }
LABEL_33:
  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  if ( (_DWORD)v82 )
    sub_2B0CE50((__int64)a1, &v81, i, v25, v19, v20);
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
  sub_C7D6A0((__int64)v78, 8LL * (unsigned int)v80, 8);
  return a1;
}
