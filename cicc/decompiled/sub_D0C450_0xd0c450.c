// Function: sub_D0C450
// Address: 0xd0c450
//
__int64 __fastcall sub_D0C450(__int64 a1, unsigned __int8 *a2, __int64 a3, char a4)
{
  unsigned int v8; // r12d
  unsigned int v10; // esi
  __int64 v11; // r8
  unsigned int v12; // edi
  unsigned __int8 **v13; // rax
  unsigned __int8 *v14; // rcx
  __int64 v15; // r10
  unsigned __int8 **v16; // r9
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // r9
  unsigned int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // rcx
  _QWORD *v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // r11
  __int64 v29; // rax
  __int64 *v30; // rax
  int v31; // ecx
  int v32; // edx
  __int64 v33; // rax
  unsigned __int64 v34; // r11
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // eax
  int v39; // edx
  __int64 v40; // rsi
  unsigned int v41; // eax
  unsigned __int8 *v42; // rdi
  int v43; // r10d
  unsigned __int8 **v44; // r8
  int v45; // eax
  int v46; // eax
  __int64 v47; // rdi
  int v48; // r10d
  unsigned int v49; // edx
  unsigned __int8 *v50; // rsi
  int v51; // eax
  int v52; // r11d
  __int64 v53; // r8
  unsigned int v54; // ecx
  __int64 v55; // rsi
  __int64 *v56; // rdi
  int v57; // eax
  int v58; // r11d
  __int64 v59; // r8
  unsigned int v60; // ecx
  __int64 v61; // rsi
  unsigned __int64 v62; // [rsp+8h] [rbp-58h]
  unsigned __int64 v63; // [rsp+10h] [rbp-50h]
  _QWORD *v64; // [rsp+10h] [rbp-50h]
  int v65; // [rsp+18h] [rbp-48h]
  _QWORD *v66; // [rsp+18h] [rbp-48h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+18h] [rbp-48h]
  __int64 v69; // [rsp+18h] [rbp-48h]
  unsigned int v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+20h] [rbp-40h]
  __int64 v72; // [rsp+20h] [rbp-40h]
  __int64 v73; // [rsp+20h] [rbp-40h]
  __int64 v74; // [rsp+20h] [rbp-40h]
  __int64 v75; // [rsp+20h] [rbp-40h]
  int v76; // [rsp+28h] [rbp-38h]
  unsigned __int8 **v77; // [rsp+28h] [rbp-38h]
  unsigned int v78; // [rsp+28h] [rbp-38h]
  __int64 v79; // [rsp+28h] [rbp-38h]
  unsigned int v80; // [rsp+28h] [rbp-38h]
  unsigned __int64 v81; // [rsp+28h] [rbp-38h]
  __int64 v82; // [rsp+28h] [rbp-38h]
  int v83; // [rsp+28h] [rbp-38h]
  int v84; // [rsp+28h] [rbp-38h]
  unsigned __int64 v85; // [rsp+28h] [rbp-38h]

  v8 = sub_CF70D0(a2);
  if ( !(_BYTE)v8 )
    return v8;
  v10 = *(_DWORD *)(a1 + 48);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_46;
  }
  v11 = *(_QWORD *)(a1 + 32);
  v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (unsigned __int8 **)(v11 + 16LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
  {
LABEL_5:
    v15 = (__int64)v13[1];
    goto LABEL_6;
  }
  v76 = 1;
  v16 = 0;
  while ( v14 != (unsigned __int8 *)-4096LL )
  {
    if ( v14 == (unsigned __int8 *)-8192LL && !v16 )
      v16 = v13;
    v12 = (v10 - 1) & (v76 + v12);
    v13 = (unsigned __int8 **)(v11 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
      goto LABEL_5;
    ++v76;
  }
  if ( !v16 )
    v16 = v13;
  v17 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v10 )
  {
LABEL_46:
    sub_D0C270(a1 + 24, 2 * v10);
    v38 = *(_DWORD *)(a1 + 48);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 32);
      v41 = (v38 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 40) + 1;
      v16 = (unsigned __int8 **)(v40 + 16LL * v41);
      v42 = *v16;
      if ( a2 == *v16 )
        goto LABEL_19;
      v43 = 1;
      v44 = 0;
      while ( v42 != (unsigned __int8 *)-4096LL )
      {
        if ( !v44 && v42 == (unsigned __int8 *)-8192LL )
          v44 = v16;
        v41 = v39 & (v43 + v41);
        v16 = (unsigned __int8 **)(v40 + 16LL * v41);
        v42 = *v16;
        if ( a2 == *v16 )
          goto LABEL_19;
        ++v43;
      }
      goto LABEL_50;
    }
LABEL_98:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( v10 - *(_DWORD *)(a1 + 44) - v18 > v10 >> 3 )
    goto LABEL_19;
  v80 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  sub_D0C270(a1 + 24, v10);
  v45 = *(_DWORD *)(a1 + 48);
  if ( !v45 )
    goto LABEL_98;
  v46 = v45 - 1;
  v47 = *(_QWORD *)(a1 + 32);
  v48 = 1;
  v44 = 0;
  v49 = v46 & v80;
  v18 = *(_DWORD *)(a1 + 40) + 1;
  v16 = (unsigned __int8 **)(v47 + 16LL * (v46 & v80));
  v50 = *v16;
  if ( a2 == *v16 )
    goto LABEL_19;
  while ( v50 != (unsigned __int8 *)-4096LL )
  {
    if ( !v44 && v50 == (unsigned __int8 *)-8192LL )
      v44 = v16;
    v49 = v46 & (v48 + v49);
    v16 = (unsigned __int8 **)(v47 + 16LL * v49);
    v50 = *v16;
    if ( a2 == *v16 )
      goto LABEL_19;
    ++v48;
  }
LABEL_50:
  if ( v44 )
    v16 = v44;
LABEL_19:
  *(_DWORD *)(a1 + 40) = v18;
  if ( *v16 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(a1 + 44);
  *v16 = a2;
  v16[1] = 0;
  v77 = v16;
  v19 = sub_D14080(a2, *(_QWORD *)(***(_QWORD ***)(a1 + 8) + 72LL), 0, *(_QWORD *)(a1 + 8), 0);
  v20 = (__int64)v77;
  v15 = v19;
  if ( !v19 )
    goto LABEL_29;
  v21 = *(_DWORD *)(a1 + 80);
  v22 = a1 + 56;
  if ( !v21 )
  {
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_62;
  }
  v23 = *(_QWORD *)(a1 + 64);
  v78 = ((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4);
  v70 = (v21 - 1) & v78;
  v24 = (_QWORD *)(v23 + 16LL * v70);
  v25 = *v24;
  if ( v19 != *v24 )
  {
    v65 = 1;
    v30 = 0;
    while ( v25 != -4096 )
    {
      if ( !v30 && v25 == -8192 )
        v30 = v24;
      v70 = (v21 - 1) & (v65 + v70);
      v24 = (_QWORD *)(v23 + 16LL * v70);
      v25 = *v24;
      if ( v15 == *v24 )
        goto LABEL_24;
      ++v65;
    }
    v31 = *(_DWORD *)(a1 + 72);
    if ( !v30 )
      v30 = v24;
    ++*(_QWORD *)(a1 + 56);
    v32 = v31 + 1;
    if ( 4 * (v31 + 1) < 3 * v21 )
    {
      if ( v21 - *(_DWORD *)(a1 + 76) - v32 > v21 >> 3 )
      {
LABEL_36:
        *(_DWORD *)(a1 + 72) = v32;
        if ( *v30 != -4096 )
          --*(_DWORD *)(a1 + 76);
        *v30 = v15;
        v27 = (unsigned __int64 *)(v30 + 1);
        *v27 = 0;
        goto LABEL_39;
      }
      v68 = v15;
      v74 = v20;
      sub_D0AD50(v22, v21);
      v57 = *(_DWORD *)(a1 + 80);
      if ( v57 )
      {
        v58 = v57 - 1;
        v59 = *(_QWORD *)(a1 + 64);
        v15 = v68;
        v60 = (v57 - 1) & v78;
        v20 = v74;
        v32 = *(_DWORD *)(a1 + 72) + 1;
        v30 = (__int64 *)(v59 + 16LL * v60);
        v61 = *v30;
        if ( v68 == *v30 )
          goto LABEL_36;
        v84 = 1;
        v56 = 0;
        while ( v61 != -4096 )
        {
          if ( !v56 && v61 == -8192 )
            v56 = v30;
          v60 = v58 & (v84 + v60);
          v30 = (__int64 *)(v59 + 16LL * v60);
          v61 = *v30;
          if ( v68 == *v30 )
            goto LABEL_36;
          ++v84;
        }
        goto LABEL_66;
      }
      goto LABEL_99;
    }
LABEL_62:
    v73 = v15;
    v82 = v20;
    sub_D0AD50(v22, 2 * v21);
    v51 = *(_DWORD *)(a1 + 80);
    if ( v51 )
    {
      v15 = v73;
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 64);
      v20 = v82;
      v32 = *(_DWORD *)(a1 + 72) + 1;
      v54 = (v51 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
      v30 = (__int64 *)(v53 + 16LL * v54);
      v55 = *v30;
      if ( v73 == *v30 )
        goto LABEL_36;
      v83 = 1;
      v56 = 0;
      while ( v55 != -4096 )
      {
        if ( !v56 && v55 == -8192 )
          v56 = v30;
        v54 = v52 & (v83 + v54);
        v30 = (__int64 *)(v53 + 16LL * v54);
        v55 = *v30;
        if ( v73 == *v30 )
          goto LABEL_36;
        ++v83;
      }
LABEL_66:
      if ( v56 )
        v30 = v56;
      goto LABEL_36;
    }
LABEL_99:
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
LABEL_24:
  v26 = v24[1];
  v27 = v24 + 1;
  v28 = v26 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v26 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v26 & 4) == 0 )
    {
      v63 = v24[1] & 0xFFFFFFFFFFFFFFF8LL;
      v66 = v24;
      v71 = v15;
      v79 = v20;
      v33 = sub_22077B0(48);
      v20 = v79;
      v15 = v71;
      v24 = v66;
      v34 = v63;
      if ( v33 )
      {
        *(_QWORD *)v33 = v33 + 16;
        *(_QWORD *)(v33 + 8) = 0x400000000LL;
      }
      v35 = v33;
      v36 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      v66[1] = v35 | 4;
      v37 = *(unsigned int *)(v36 + 8);
      if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 12) )
      {
        v62 = v63;
        v64 = v66;
        v69 = v71;
        v75 = v79;
        v85 = v36;
        sub_C8D5F0(v36, (const void *)(v36 + 16), v37 + 1, 8u, (__int64)v24, v20);
        v36 = v85;
        v34 = v62;
        v24 = v64;
        v15 = v69;
        v37 = *(unsigned int *)(v85 + 8);
        v20 = v75;
      }
      *(_QWORD *)(*(_QWORD *)v36 + 8 * v37) = v34;
      ++*(_DWORD *)(v36 + 8);
      v28 = v24[1] & 0xFFFFFFFFFFFFFFF8LL;
    }
    v29 = *(unsigned int *)(v28 + 8);
    if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 12) )
    {
      v67 = v15;
      v72 = v20;
      v81 = v28;
      sub_C8D5F0(v28, (const void *)(v28 + 16), v29 + 1, 8u, (__int64)v24, v20);
      v28 = v81;
      v15 = v67;
      v20 = v72;
      v29 = *(unsigned int *)(v81 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v28 + 8 * v29) = a2;
    ++*(_DWORD *)(v28 + 8);
    goto LABEL_29;
  }
LABEL_39:
  *v27 = (unsigned __int64)a2 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_29:
  *(_QWORD *)(v20 + 8) = v15;
LABEL_6:
  if ( !v15 )
    return v8;
  if ( !a3 )
    return 0;
  if ( a3 != v15 )
    return (unsigned int)sub_D0EBA0(v15, a3, 0, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16)) ^ 1;
  v8 = 0;
  if ( a4 )
    return v8;
  return sub_D00920(a3, *(_BYTE **)(a1 + 8), *(_QWORD *)(a1 + 16));
}
