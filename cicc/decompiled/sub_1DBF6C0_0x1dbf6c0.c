// Function: sub_1DBF6C0
// Address: 0x1dbf6c0
//
__int64 __fastcall sub_1DBF6C0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, int *a5, __int64 a6)
{
  __int64 v7; // r12
  unsigned __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // rdi
  int v11; // ecx
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  int v14; // r9d
  unsigned int v15; // edx
  unsigned __int64 v16; // r10
  unsigned int v17; // edx
  __int64 v18; // r8
  unsigned int v19; // r11d
  unsigned int v20; // eax
  __int64 v21; // r9
  unsigned __int64 j; // rax
  unsigned int v23; // r9d
  __int64 *v24; // rcx
  __int64 v25; // r13
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r13
  int v28; // r9d
  unsigned __int64 v29; // r8
  __int64 v30; // rbx
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 k; // r13
  __int64 v35; // r14
  __int64 v36; // r8
  int v37; // r15d
  unsigned __int64 v38; // rcx
  unsigned int v39; // eax
  unsigned int v40; // eax
  __int64 v41; // rsi
  int *v42; // r13
  __int64 v43; // r10
  unsigned __int64 v44; // rbx
  __int64 result; // rax
  __int64 v46; // r15
  int v47; // r14d
  unsigned __int64 v48; // rdx
  unsigned int v49; // eax
  __int64 v50; // r8
  __int64 v51; // r11
  __int64 v52; // r13
  __int64 v53; // r12
  int v54; // r13d
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  int v57; // edx
  int v58; // r14d
  __int64 i; // rax
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rcx
  __int64 v63; // r9
  _QWORD *v64; // rdi
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rdi
  _QWORD *v68; // rsi
  _QWORD *v69; // rdx
  int v70; // ecx
  int v71; // r14d
  __int64 v72; // [rsp+0h] [rbp-80h]
  unsigned int v73; // [rsp+Ch] [rbp-74h]
  unsigned int v76; // [rsp+20h] [rbp-60h]
  const void *v77; // [rsp+28h] [rbp-58h]
  unsigned __int64 v78; // [rsp+30h] [rbp-50h]
  int *v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  __int64 v82; // [rsp+38h] [rbp-48h]
  unsigned __int64 v83; // [rsp+40h] [rbp-40h]
  __int64 v84; // [rsp+40h] [rbp-40h]
  int *v85; // [rsp+40h] [rbp-40h]
  __int64 v86; // [rsp+40h] [rbp-40h]
  __int64 v87; // [rsp+48h] [rbp-38h]

  v7 = a1;
  v8 = a4;
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_QWORD *)(a1 + 272);
  if ( a3 == v9 )
  {
    v16 = a3;
  }
  else
  {
    v11 = *(_DWORD *)(v10 + 384);
    v12 = *(_QWORD *)(v10 + 368);
    v13 = a3;
    v14 = v11 - 1;
    do
    {
      if ( v11 )
      {
        v15 = v14 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v16 = *(_QWORD *)(v12 + 16LL * v15);
        if ( v13 == v16 )
          goto LABEL_5;
        v58 = 1;
        while ( v16 != -8 )
        {
          v15 = v14 & (v58 + v15);
          v16 = *(_QWORD *)(v12 + 16LL * v15);
          if ( v13 == v16 )
            goto LABEL_5;
          ++v58;
        }
      }
      v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v13 )
        BUG();
      if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v13; ; i = *(_QWORD *)v13 )
        {
          v13 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v13 + 46) & 4) == 0 )
            break;
        }
      }
    }
    while ( v13 != v9 );
    v16 = v13;
  }
LABEL_5:
  if ( a4 == a2 + 24 )
  {
LABEL_51:
    v55 = *(_QWORD *)(*(_QWORD *)(v10 + 392) + 16LL * *(unsigned int *)(a2 + 48) + 8);
    v56 = v55 & 0xFFFFFFFFFFFFFFF8LL;
    v57 = (v55 >> 1) & 3;
    if ( v57 )
      v26 = (2LL * (v57 - 1)) | v56;
    else
      v26 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL | 6;
    goto LABEL_13;
  }
  v17 = *(_DWORD *)(v10 + 384);
  v18 = *(_QWORD *)(v10 + 368);
  v19 = v17 - 1;
  while ( !v17 )
  {
LABEL_48:
    if ( !v8 )
      BUG();
    if ( (*(_BYTE *)v8 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
        v8 = *(_QWORD *)(v8 + 8);
    }
    v8 = *(_QWORD *)(v8 + 8);
    if ( a2 + 24 == v8 )
      goto LABEL_51;
  }
  v20 = v19 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v21 = *(_QWORD *)(v18 + 16LL * v20);
  if ( v8 != v21 )
  {
    v54 = 1;
    while ( v21 != -8 )
    {
      v20 = v19 & (v54 + v20);
      v21 = *(_QWORD *)(v18 + 16LL * v20);
      if ( v8 == v21 )
        goto LABEL_9;
      ++v54;
    }
    goto LABEL_48;
  }
LABEL_9:
  for ( j = v8; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v23 = v19 & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
  v24 = (__int64 *)(v18 + 16LL * v23);
  v25 = *v24;
  if ( j != *v24 )
  {
    v70 = 1;
    while ( v25 != -8 )
    {
      v71 = v70 + 1;
      v23 = v19 & (v70 + v23);
      v24 = (__int64 *)(v18 + 16LL * v23);
      v25 = *v24;
      if ( j == *v24 )
        goto LABEL_12;
      v70 = v71;
    }
    v24 = (__int64 *)(v18 + 16LL * v17);
  }
LABEL_12:
  v26 = v24[1];
LABEL_13:
  v87 = v26;
  v27 = v8;
  v83 = v16;
  sub_1F10BF0(v10, a2, v16, v8);
  v29 = v8;
  v30 = v83;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( v30 == v27 )
        goto LABEL_33;
LABEL_15:
      v31 = (_QWORD *)(*(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL);
      v32 = v31;
      if ( !v31 )
        BUG();
      v27 = *(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL;
      v33 = *v31;
      if ( (v33 & 4) == 0 && (*((_BYTE *)v32 + 46) & 4) != 0 )
      {
        for ( k = v33; ; k = *(_QWORD *)v27 )
        {
          v27 = k & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v27 + 46) & 4) == 0 )
            break;
        }
      }
      if ( (unsigned __int16)(**(_WORD **)(v27 + 16) - 12) <= 1u )
        continue;
      break;
    }
    v35 = *(_QWORD *)(v27 + 32);
    if ( v35 + 40LL * *(unsigned int *)(v27 + 40) == v35 )
      continue;
    break;
  }
  v78 = v29;
  v36 = v35 + 40LL * *(unsigned int *)(v27 + 40);
  while ( 2 )
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v35 )
      {
        v37 = *(_DWORD *)(v35 + 8);
        if ( v37 < 0 )
        {
          v38 = *(unsigned int *)(v7 + 408);
          v39 = v37 & 0x7FFFFFFF;
          if ( (v37 & 0x7FFFFFFFu) >= (unsigned int)v38 || !*(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v39) )
            break;
        }
      }
      v35 += 40;
      if ( v35 == v36 )
        goto LABEL_32;
    }
    v40 = v39 + 1;
    if ( (unsigned int)v38 >= v40 )
      goto LABEL_30;
    v60 = v40;
    if ( v40 < v38 )
    {
      *(_DWORD *)(v7 + 408) = v40;
      goto LABEL_30;
    }
    if ( v40 <= v38 )
    {
LABEL_30:
      v41 = *(_QWORD *)(v7 + 400);
    }
    else
    {
      if ( v40 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
      {
        v72 = v36;
        v73 = v40;
        v86 = v40;
        sub_16CD150(v7 + 400, (const void *)(v7 + 416), v40, 8, v36, v28);
        v38 = *(unsigned int *)(v7 + 408);
        v36 = v72;
        v40 = v73;
        v60 = v86;
      }
      v41 = *(_QWORD *)(v7 + 400);
      v64 = (_QWORD *)(v41 + 8 * v60);
      v65 = (_QWORD *)(v41 + 8 * v38);
      v66 = *(_QWORD *)(v7 + 416);
      if ( v64 != v65 )
      {
        do
          *v65++ = v66;
        while ( v64 != v65 );
        v41 = *(_QWORD *)(v7 + 400);
      }
      *(_DWORD *)(v7 + 408) = v40;
    }
    v84 = v36;
    v35 += 40;
    *(_QWORD *)(v41 + 8LL * (v37 & 0x7FFFFFFF)) = sub_1DBA290(v37);
    sub_1DBB110((_QWORD *)v7, *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * (v37 & 0x7FFFFFFF)));
    v36 = v84;
    if ( v35 != v84 )
      continue;
    break;
  }
LABEL_32:
  v29 = v78;
  if ( v30 != v27 )
    goto LABEL_15;
LABEL_33:
  v42 = a5;
  v43 = v30;
  v77 = (const void *)(v7 + 416);
  v44 = v29;
  result = (__int64)&a5[a6];
  v85 = (int *)result;
  if ( a5 != (int *)result )
  {
    v46 = v43;
    while ( 2 )
    {
      while ( 1 )
      {
        v47 = *v42;
        if ( *v42 < 0 )
          break;
LABEL_35:
        if ( v85 == ++v42 )
          return result;
      }
      v48 = *(unsigned int *)(v7 + 408);
      v49 = v47 & 0x7FFFFFFF;
      v50 = v47 & 0x7FFFFFFF;
      v51 = 8 * v50;
      if ( (v47 & 0x7FFFFFFFu) < (unsigned int)v48 )
      {
        v81 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v49);
        if ( v81 )
        {
LABEL_39:
          result = v81;
          if ( !*(_DWORD *)(v81 + 72) )
            goto LABEL_35;
          if ( *(_QWORD *)(v81 + 104) )
          {
            v79 = v42;
            v52 = v7;
            v53 = *(_QWORD *)(v81 + 104);
            do
            {
              sub_1DBF150(v52, v46, v44, v87, v53, v47, *(_DWORD *)(v53 + 112));
              v53 = *(_QWORD *)(v53 + 104);
            }
            while ( v53 );
            v7 = v52;
            v42 = v79;
          }
          ++v42;
          result = sub_1DBF150(v7, v46, v44, v87, v81, v47, -1);
          if ( v85 == v42 )
            return result;
          continue;
        }
      }
      break;
    }
    v61 = v49 + 1;
    if ( (unsigned int)v48 >= v61 )
      goto LABEL_69;
    v63 = v61;
    if ( v61 < v48 )
    {
      *(_DWORD *)(v7 + 408) = v61;
      goto LABEL_69;
    }
    if ( v61 <= v48 )
    {
LABEL_69:
      v62 = *(_QWORD *)(v7 + 400);
    }
    else
    {
      if ( v61 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
      {
        v76 = v61;
        v82 = v61;
        sub_16CD150(v7 + 400, v77, v61, 8, v50, v61);
        v48 = *(unsigned int *)(v7 + 408);
        v50 = v47 & 0x7FFFFFFF;
        v61 = v76;
        v51 = 8 * v50;
        v63 = v82;
      }
      v62 = *(_QWORD *)(v7 + 400);
      v67 = *(_QWORD *)(v7 + 416);
      v68 = (_QWORD *)(v62 + 8 * v63);
      v69 = (_QWORD *)(v62 + 8 * v48);
      if ( v68 != v69 )
      {
        do
          *v69++ = v67;
        while ( v68 != v69 );
        v62 = *(_QWORD *)(v7 + 400);
      }
      *(_DWORD *)(v7 + 408) = v61;
    }
    v80 = v50;
    *(_QWORD *)(v62 + v51) = sub_1DBA290(v47);
    v81 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8 * v80);
    sub_1DBB110((_QWORD *)v7, v81);
    goto LABEL_39;
  }
  return result;
}
