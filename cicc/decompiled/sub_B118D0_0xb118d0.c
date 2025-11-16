// Function: sub_B118D0
// Address: 0xb118d0
//
_QWORD *__fastcall sub_B118D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // rbx
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  _BYTE *v11; // r15
  __int64 v12; // r8
  __int64 *v13; // rdx
  _QWORD *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // r9
  int v17; // r11d
  _QWORD *v18; // rdx
  unsigned int v19; // edi
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rbx
  unsigned __int8 v23; // al
  int v24; // eax
  int v25; // eax
  __int64 v27; // rbx
  unsigned int v28; // esi
  __int64 v29; // rcx
  int v30; // r11d
  __int64 *v31; // r8
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r10
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  int v38; // eax
  int v39; // edx
  int v40; // esi
  int v41; // esi
  __int64 v42; // r9
  unsigned int v43; // ecx
  __int64 v44; // rdi
  int v45; // r13d
  _QWORD *v46; // r10
  int v47; // ecx
  int v48; // ecx
  __int64 v49; // rdi
  _QWORD *v50; // r9
  unsigned int v51; // r13d
  int v52; // r11d
  __int64 v53; // rsi
  int v54; // eax
  int v55; // ecx
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rdi
  int v59; // r10d
  __int64 *v60; // r9
  int v61; // eax
  int v62; // eax
  __int64 v63; // rsi
  int v64; // r9d
  __int64 v65; // r15
  __int64 *v66; // rdi
  __int64 v67; // rcx
  __int64 v68; // [rsp+0h] [rbp-80h]
  __int64 v69; // [rsp+0h] [rbp-80h]
  _BYTE *v71; // [rsp+18h] [rbp-68h]
  _BYTE *v72; // [rsp+20h] [rbp-60h] BYREF
  __int64 v73; // [rsp+28h] [rbp-58h]
  _BYTE v74[80]; // [rsp+30h] [rbp-50h] BYREF

  v72 = v74;
  v73 = 0x300000000LL;
  v8 = sub_B10CD0(a2);
  while ( 1 )
  {
    v9 = *(_BYTE *)(v8 - 16);
    if ( (v9 & 2) != 0 )
    {
      if ( *(_DWORD *)(v8 - 24) != 2 )
        goto LABEL_4;
      v27 = *(_QWORD *)(v8 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_4;
      v27 = v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
    }
    v8 = *(_QWORD *)(v27 + 8);
    if ( !v8 )
    {
LABEL_4:
      v10 = (unsigned int)v73;
      goto LABEL_5;
    }
    v28 = *(_DWORD *)(a5 + 24);
    if ( !v28 )
    {
      ++*(_QWORD *)a5;
LABEL_69:
      sub_B00360(a5, 2 * v28);
      v54 = *(_DWORD *)(a5 + 24);
      if ( !v54 )
        goto LABEL_103;
      v55 = v54 - 1;
      v56 = *(_QWORD *)(a5 + 8);
      LODWORD(v57) = (v54 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v39 = *(_DWORD *)(a5 + 16) + 1;
      v31 = (__int64 *)(v56 + 16LL * (unsigned int)v57);
      v58 = *v31;
      if ( v8 != *v31 )
      {
        v59 = 1;
        v60 = 0;
        while ( v58 != -4096 )
        {
          if ( !v60 && v58 == -8192 )
            v60 = v31;
          v57 = v55 & (unsigned int)(v57 + v59);
          v31 = (__int64 *)(v56 + 16 * v57);
          v58 = *v31;
          if ( v8 == *v31 )
            goto LABEL_49;
          ++v59;
        }
        if ( v60 )
          v31 = v60;
      }
      goto LABEL_49;
    }
    v29 = *(_QWORD *)(a5 + 8);
    v30 = 1;
    v31 = 0;
    v32 = (v28 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v33 = (__int64 *)(v29 + 16LL * v32);
    v34 = *v33;
    if ( v8 == *v33 )
      break;
    while ( v34 != -4096 )
    {
      if ( !v31 && v34 == -8192 )
        v31 = v33;
      v32 = (v28 - 1) & (v30 + v32);
      v33 = (__int64 *)(v29 + 16LL * v32);
      v34 = *v33;
      if ( v8 == *v33 )
        goto LABEL_35;
      ++v30;
    }
    if ( !v31 )
      v31 = v33;
    v38 = *(_DWORD *)(a5 + 16);
    ++*(_QWORD *)a5;
    v39 = v38 + 1;
    if ( 4 * (v38 + 1) >= 3 * v28 )
      goto LABEL_69;
    if ( v28 - *(_DWORD *)(a5 + 20) - v39 <= v28 >> 3 )
    {
      sub_B00360(a5, v28);
      v61 = *(_DWORD *)(a5 + 24);
      if ( !v61 )
      {
LABEL_103:
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
      v62 = v61 - 1;
      v63 = *(_QWORD *)(a5 + 8);
      v64 = 1;
      LODWORD(v65) = v62 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v39 = *(_DWORD *)(a5 + 16) + 1;
      v66 = 0;
      v31 = (__int64 *)(v63 + 16LL * (unsigned int)v65);
      v67 = *v31;
      if ( v8 != *v31 )
      {
        while ( v67 != -4096 )
        {
          if ( v67 == -8192 && !v66 )
            v66 = v31;
          v65 = v62 & (unsigned int)(v65 + v64);
          v31 = (__int64 *)(v63 + 16 * v65);
          v67 = *v31;
          if ( v8 == *v31 )
            goto LABEL_49;
          ++v64;
        }
        if ( v66 )
          v31 = v66;
      }
    }
LABEL_49:
    *(_DWORD *)(a5 + 16) = v39;
    if ( *v31 != -4096 )
      --*(_DWORD *)(a5 + 20);
    *v31 = v8;
    v31[1] = 0;
    v36 = (unsigned int)v73;
    v37 = (unsigned int)v73 + 1LL;
    if ( v37 > HIDWORD(v73) )
      goto LABEL_52;
LABEL_37:
    *(_QWORD *)&v72[8 * v36] = v8;
    LODWORD(v73) = v73 + 1;
  }
LABEL_35:
  v35 = v33[1];
  v10 = (unsigned int)v73;
  v36 = (unsigned int)v73;
  if ( !v35 )
  {
    v37 = (unsigned int)v73 + 1LL;
    if ( v37 <= HIDWORD(v73) )
      goto LABEL_37;
LABEL_52:
    sub_C8D5F0(&v72, v74, v37, 8);
    v36 = (unsigned int)v73;
    goto LABEL_37;
  }
  a3 = v35;
LABEL_5:
  v11 = &v72[8 * v10];
  v71 = v72;
  if ( v72 == v11 )
    goto LABEL_27;
  v12 = a3;
  while ( 2 )
  {
    while ( 2 )
    {
      v22 = *((_QWORD *)v11 - 1);
      v23 = *(_BYTE *)(v22 - 16);
      if ( (v23 & 2) != 0 )
        v13 = *(__int64 **)(v22 - 32);
      else
        v13 = (__int64 *)(v22 - 16 - 8LL * ((v23 >> 2) & 0xF));
      v14 = sub_B01860(a4, *(_DWORD *)(v22 + 4), *(unsigned __int16 *)(v22 + 2), *v13, v12, 0, 1u, 1);
      v15 = *(_DWORD *)(a5 + 24);
      v12 = (__int64)v14;
      if ( !v15 )
      {
        ++*(_QWORD *)a5;
        goto LABEL_55;
      }
      v16 = *(_QWORD *)(a5 + 8);
      v17 = 1;
      v18 = 0;
      v19 = (v15 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v20 = (_QWORD *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v22 == *v20 )
      {
LABEL_10:
        v11 -= 8;
        v20[1] = v12;
        if ( v71 == v11 )
          goto LABEL_26;
        continue;
      }
      break;
    }
    while ( v21 != -4096 )
    {
      if ( v21 == -8192 && !v18 )
        v18 = v20;
      v19 = (v15 - 1) & (v17 + v19);
      v20 = (_QWORD *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v22 == *v20 )
        goto LABEL_10;
      ++v17;
    }
    if ( !v18 )
      v18 = v20;
    v24 = *(_DWORD *)(a5 + 16);
    ++*(_QWORD *)a5;
    v25 = v24 + 1;
    if ( 4 * v25 >= 3 * v15 )
    {
LABEL_55:
      v68 = v12;
      sub_B00360(a5, 2 * v15);
      v40 = *(_DWORD *)(a5 + 24);
      if ( !v40 )
        goto LABEL_102;
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a5 + 8);
      v12 = v68;
      v43 = v41 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v25 = *(_DWORD *)(a5 + 16) + 1;
      v18 = (_QWORD *)(v42 + 16LL * v43);
      v44 = *v18;
      if ( v22 != *v18 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != -4096 )
        {
          if ( v44 == -8192 && !v46 )
            v46 = v18;
          v43 = v41 & (v45 + v43);
          v18 = (_QWORD *)(v42 + 16LL * v43);
          v44 = *v18;
          if ( v22 == *v18 )
            goto LABEL_23;
          ++v45;
        }
        if ( v46 )
          v18 = v46;
      }
      goto LABEL_23;
    }
    if ( v15 - *(_DWORD *)(a5 + 20) - v25 <= v15 >> 3 )
    {
      v69 = v12;
      sub_B00360(a5, v15);
      v47 = *(_DWORD *)(a5 + 24);
      if ( !v47 )
      {
LABEL_102:
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a5 + 8);
      v50 = 0;
      v51 = v48 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v12 = v69;
      v52 = 1;
      v25 = *(_DWORD *)(a5 + 16) + 1;
      v18 = (_QWORD *)(v49 + 16LL * v51);
      v53 = *v18;
      if ( v22 != *v18 )
      {
        while ( v53 != -4096 )
        {
          if ( !v50 && v53 == -8192 )
            v50 = v18;
          v51 = v48 & (v52 + v51);
          v18 = (_QWORD *)(v49 + 16LL * v51);
          v53 = *v18;
          if ( v22 == *v18 )
            goto LABEL_23;
          ++v52;
        }
        if ( v50 )
          v18 = v50;
      }
    }
LABEL_23:
    *(_DWORD *)(a5 + 16) = v25;
    if ( *v18 != -4096 )
      --*(_DWORD *)(a5 + 20);
    *v18 = v22;
    v11 -= 8;
    v18[1] = 0;
    v18[1] = v12;
    if ( v71 != v11 )
      continue;
    break;
  }
LABEL_26:
  a3 = v12;
LABEL_27:
  sub_B10CB0(a1, a3);
  if ( v72 != v74 )
    _libc_free(v72, a3);
  return a1;
}
