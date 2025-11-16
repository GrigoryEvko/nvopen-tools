// Function: sub_1416A50
// Address: 0x1416a50
//
__int64 __fastcall sub_1416A50(__int64 a1)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rbx
  char v5; // cl
  char v6; // r9
  __int64 v7; // rsi
  int v8; // r10d
  unsigned int v9; // r8d
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 *v14; // r12
  __int64 *v15; // r9
  __int64 v16; // rdi
  int v17; // esi
  unsigned int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r13
  int v22; // ecx
  unsigned int v23; // esi
  unsigned int v24; // edx
  int v25; // edi
  unsigned int v26; // r8d
  __int64 v27; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // r11d
  _QWORD *v32; // r10
  __int64 *v33; // r12
  __int64 *v34; // r15
  __int64 v35; // rdi
  int v36; // esi
  unsigned int v37; // edx
  _QWORD *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r13
  char v41; // cl
  unsigned int v42; // esi
  int v43; // eax
  __int64 v44; // rcx
  int v45; // esi
  unsigned int v46; // edx
  __int64 v47; // rdi
  int v48; // ecx
  __int64 v49; // rsi
  int v50; // ecx
  unsigned int v51; // edx
  __int64 v52; // rdi
  int v53; // r10d
  _QWORD *v54; // r8
  __int64 v55; // rax
  unsigned int v56; // edx
  int v57; // edi
  unsigned int v58; // r8d
  int v59; // r10d
  _QWORD *v60; // r9
  int v61; // eax
  int v62; // ecx
  __int64 v63; // rdi
  int v64; // ecx
  unsigned int v65; // edx
  __int64 v66; // rsi
  int v67; // ecx
  __int64 v68; // rsi
  int v69; // ecx
  unsigned int v70; // edx
  __int64 v71; // rdi
  int v72; // r9d
  _QWORD *v73; // r8
  int v74; // r10d
  int v75; // r11d
  int v76; // r9d
  __int64 *v77; // [rsp+8h] [rbp-38h]
  __int64 *v78; // [rsp+8h] [rbp-38h]

  v2 = **(_QWORD **)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a1 + 16) = v2;
  v3 = 0;
  v4 = *(_QWORD *)a1;
  if ( v2 )
    v3 = v2 - 24;
  *(_QWORD *)(a1 + 24) = v3;
  v5 = *(_BYTE *)(v4 + 8);
  v6 = v5 & 1;
  if ( (v5 & 1) != 0 )
  {
    v7 = v4 + 16;
    v8 = 3;
  }
  else
  {
    v29 = *(unsigned int *)(v4 + 24);
    v7 = *(_QWORD *)(v4 + 16);
    if ( !(_DWORD)v29 )
      goto LABEL_32;
    v8 = v29 - 1;
  }
  v9 = v8 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v3 == *v10 )
    goto LABEL_6;
  v61 = 1;
  while ( v11 != -8 )
  {
    v75 = v61 + 1;
    v9 = v8 & (v61 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
      goto LABEL_6;
    v61 = v75;
  }
  if ( v6 )
  {
    v30 = 64;
    goto LABEL_33;
  }
  v29 = *(unsigned int *)(v4 + 24);
LABEL_32:
  v30 = 16 * v29;
LABEL_33:
  v10 = (__int64 *)(v7 + v30);
LABEL_6:
  v12 = 64;
  if ( !v6 )
    v12 = 16LL * *(unsigned int *)(v4 + 24);
  if ( v10 != (__int64 *)(v7 + v12) )
  {
    v13 = v10[1];
    v14 = *(__int64 **)(a1 + 32);
    v15 = &v14[*(unsigned int *)(a1 + 40)];
    if ( v14 == v15 )
    {
LABEL_25:
      *(_DWORD *)(a1 + 40) = 0;
      if ( v13 )
        v27 = v13 + 24;
      else
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
      *(_QWORD *)(a1 + 16) = v27;
      return a1;
    }
    while ( 1 )
    {
      v21 = *v14;
      v22 = v5 & 1;
      if ( v22 )
      {
        v16 = v4 + 16;
        v17 = 3;
      }
      else
      {
        v23 = *(_DWORD *)(v4 + 24);
        v16 = *(_QWORD *)(v4 + 16);
        if ( !v23 )
        {
          v24 = *(_DWORD *)(v4 + 8);
          ++*(_QWORD *)v4;
          v19 = 0;
          v25 = (v24 >> 1) + 1;
LABEL_19:
          v26 = 3 * v23;
          goto LABEL_20;
        }
        v17 = v23 - 1;
      }
      v18 = v17 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v19 = (_QWORD *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( v21 == *v19 )
      {
LABEL_13:
        ++v14;
        v19[1] = v13;
        if ( v15 == v14 )
          goto LABEL_25;
        goto LABEL_14;
      }
      v31 = 1;
      v32 = 0;
      while ( v20 != -8 )
      {
        if ( !v32 && v20 == -16 )
          v32 = v19;
        v18 = v17 & (v31 + v18);
        v19 = (_QWORD *)(v16 + 16LL * v18);
        v20 = *v19;
        if ( v21 == *v19 )
          goto LABEL_13;
        ++v31;
      }
      v24 = *(_DWORD *)(v4 + 8);
      v26 = 12;
      v23 = 4;
      if ( v32 )
        v19 = v32;
      ++*(_QWORD *)v4;
      v25 = (v24 >> 1) + 1;
      if ( !(_BYTE)v22 )
      {
        v23 = *(_DWORD *)(v4 + 24);
        goto LABEL_19;
      }
LABEL_20:
      if ( v26 <= 4 * v25 )
      {
        v77 = v15;
        sub_14163A0(v4, 2 * v23);
        v15 = v77;
        if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
        {
          v44 = v4 + 16;
          v45 = 3;
        }
        else
        {
          v43 = *(_DWORD *)(v4 + 24);
          v44 = *(_QWORD *)(v4 + 16);
          if ( !v43 )
            goto LABEL_137;
          v45 = v43 - 1;
        }
        v46 = v45 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v19 = (_QWORD *)(v44 + 16LL * v46);
        v47 = *v19;
        if ( v21 == *v19 )
          goto LABEL_54;
        v74 = 1;
        v54 = 0;
        while ( v47 != -8 )
        {
          if ( v47 == -16 && !v54 )
            v54 = v19;
          v46 = v45 & (v74 + v46);
          v19 = (_QWORD *)(v44 + 16LL * v46);
          v47 = *v19;
          if ( v21 == *v19 )
            goto LABEL_54;
          ++v74;
        }
        goto LABEL_61;
      }
      if ( v23 - *(_DWORD *)(v4 + 12) - v25 > v23 >> 3 )
        goto LABEL_22;
      v78 = v15;
      sub_14163A0(v4, v23);
      v15 = v78;
      if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
      {
        v49 = v4 + 16;
        v50 = 3;
      }
      else
      {
        v48 = *(_DWORD *)(v4 + 24);
        v49 = *(_QWORD *)(v4 + 16);
        if ( !v48 )
          goto LABEL_137;
        v50 = v48 - 1;
      }
      v51 = v50 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v19 = (_QWORD *)(v49 + 16LL * v51);
      v52 = *v19;
      if ( v21 != *v19 )
      {
        v53 = 1;
        v54 = 0;
        while ( v52 != -8 )
        {
          if ( !v54 && v52 == -16 )
            v54 = v19;
          v51 = v50 & (v53 + v51);
          v19 = (_QWORD *)(v49 + 16LL * v51);
          v52 = *v19;
          if ( v21 == *v19 )
            goto LABEL_54;
          ++v53;
        }
LABEL_61:
        if ( v54 )
          v19 = v54;
      }
LABEL_54:
      v24 = *(_DWORD *)(v4 + 8);
LABEL_22:
      *(_DWORD *)(v4 + 8) = (2 * (v24 >> 1) + 2) | v24 & 1;
      if ( *v19 != -8 )
        --*(_DWORD *)(v4 + 12);
      ++v14;
      v19[1] = 0;
      *v19 = v21;
      v19[1] = v13;
      if ( v15 == v14 )
        goto LABEL_25;
LABEL_14:
      v4 = *(_QWORD *)a1;
      v5 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    }
  }
  if ( (unsigned __int8)sub_1412690(v3) )
  {
    v33 = *(__int64 **)(a1 + 32);
    v34 = &v33[*(unsigned int *)(a1 + 40)];
    if ( v33 == v34 )
    {
LABEL_75:
      *(_DWORD *)(a1 + 40) = 0;
      sub_1416780(*(_QWORD *)a1, (__int64 *)(a1 + 24))[1] = *(_QWORD *)(a1 + 24);
      return a1;
    }
    while ( 1 )
    {
      v4 = *(_QWORD *)a1;
      v40 = *v33;
      v41 = *(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1;
      if ( v41 )
      {
        v35 = v4 + 16;
        v36 = 3;
      }
      else
      {
        v42 = *(_DWORD *)(v4 + 24);
        v35 = *(_QWORD *)(v4 + 16);
        if ( !v42 )
        {
          v56 = *(_DWORD *)(v4 + 8);
          ++*(_QWORD *)v4;
          v38 = 0;
          v57 = (v56 >> 1) + 1;
          goto LABEL_69;
        }
        v36 = v42 - 1;
      }
      v37 = v36 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v38 = (_QWORD *)(v35 + 16LL * v37);
      v39 = *v38;
      if ( *v38 != v40 )
        break;
LABEL_46:
      ++v33;
      v38[1] = *(_QWORD *)(a1 + 24);
      if ( v34 == v33 )
        goto LABEL_75;
    }
    v59 = 1;
    v60 = 0;
    while ( v39 != -8 )
    {
      if ( !v60 && v39 == -16 )
        v60 = v38;
      v37 = v36 & (v59 + v37);
      v38 = (_QWORD *)(v35 + 16LL * v37);
      v39 = *v38;
      if ( v40 == *v38 )
        goto LABEL_46;
      ++v59;
    }
    v56 = *(_DWORD *)(v4 + 8);
    v58 = 12;
    v42 = 4;
    if ( v60 )
      v38 = v60;
    ++*(_QWORD *)v4;
    v57 = (v56 >> 1) + 1;
    if ( !v41 )
    {
      v42 = *(_DWORD *)(v4 + 24);
LABEL_69:
      v58 = 3 * v42;
    }
    if ( 4 * v57 >= v58 )
    {
      sub_14163A0(v4, 2 * v42);
      if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
      {
        v63 = v4 + 16;
        v64 = 3;
      }
      else
      {
        v62 = *(_DWORD *)(v4 + 24);
        v63 = *(_QWORD *)(v4 + 16);
        if ( !v62 )
          goto LABEL_137;
        v64 = v62 - 1;
      }
      v65 = v64 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v38 = (_QWORD *)(v63 + 16LL * v65);
      v66 = *v38;
      if ( *v38 == v40 )
        goto LABEL_90;
      v76 = 1;
      v73 = 0;
      while ( v66 != -8 )
      {
        if ( v66 == -16 && !v73 )
          v73 = v38;
        v65 = v64 & (v76 + v65);
        v38 = (_QWORD *)(v63 + 16LL * v65);
        v66 = *v38;
        if ( v40 == *v38 )
          goto LABEL_90;
        ++v76;
      }
    }
    else
    {
      if ( v42 - *(_DWORD *)(v4 + 12) - v57 > v42 >> 3 )
      {
LABEL_72:
        *(_DWORD *)(v4 + 8) = (2 * (v56 >> 1) + 2) | v56 & 1;
        if ( *v38 != -8 )
          --*(_DWORD *)(v4 + 12);
        *v38 = v40;
        v38[1] = 0;
        goto LABEL_46;
      }
      sub_14163A0(v4, v42);
      if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
      {
        v68 = v4 + 16;
        v69 = 3;
      }
      else
      {
        v67 = *(_DWORD *)(v4 + 24);
        v68 = *(_QWORD *)(v4 + 16);
        if ( !v67 )
        {
LABEL_137:
          *(_DWORD *)(v4 + 8) = (2 * (*(_DWORD *)(v4 + 8) >> 1) + 2) | *(_DWORD *)(v4 + 8) & 1;
          BUG();
        }
        v69 = v67 - 1;
      }
      v70 = v69 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v38 = (_QWORD *)(v68 + 16LL * v70);
      v71 = *v38;
      if ( *v38 == v40 )
      {
LABEL_90:
        v56 = *(_DWORD *)(v4 + 8);
        goto LABEL_72;
      }
      v72 = 1;
      v73 = 0;
      while ( v71 != -8 )
      {
        if ( !v73 && v71 == -16 )
          v73 = v38;
        v70 = v69 & (v72 + v70);
        v38 = (_QWORD *)(v68 + 16LL * v70);
        v71 = *v38;
        if ( v40 == *v38 )
          goto LABEL_90;
        ++v72;
      }
    }
    if ( v73 )
      v38 = v73;
    goto LABEL_90;
  }
  v55 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v55 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, a1 + 48, 0, 8);
    v55 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v55) = *(_QWORD *)(a1 + 24);
  ++*(_DWORD *)(a1 + 40);
  return a1;
}
