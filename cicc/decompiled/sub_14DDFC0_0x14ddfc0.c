// Function: sub_14DDFC0
// Address: 0x14ddfc0
//
__int64 __fastcall sub_14DDFC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  _QWORD *v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rbx
  unsigned int v12; // ecx
  unsigned int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // r8
  char v21; // si
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rdi
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // r8
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r15d
  int v41; // r12d
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 *v44; // rax
  __int64 v45; // r8
  __int64 v46; // rax
  int v47; // r10d
  __int64 *v48; // r9
  int v49; // ecx
  int v50; // ecx
  int v51; // eax
  int v52; // esi
  __int64 v53; // rdx
  unsigned int v54; // eax
  __int64 v55; // rdi
  int v56; // r9d
  __int64 *v57; // r8
  int v58; // edx
  int v59; // edx
  __int64 v60; // rdi
  int v61; // r9d
  unsigned int v62; // eax
  __int64 v63; // rsi
  unsigned __int64 v64; // [rsp+8h] [rbp-168h]
  __int64 v65; // [rsp+10h] [rbp-160h]
  unsigned __int64 v66; // [rsp+20h] [rbp-150h]
  __int64 v67; // [rsp+20h] [rbp-150h]
  unsigned int v68; // [rsp+20h] [rbp-150h]
  unsigned __int64 v69; // [rsp+20h] [rbp-150h]
  _QWORD *v70; // [rsp+30h] [rbp-140h] BYREF
  unsigned int v71; // [rsp+38h] [rbp-138h]
  unsigned int v72; // [rsp+3Ch] [rbp-134h]
  _QWORD v73[38]; // [rsp+40h] [rbp-130h] BYREF

  v2 = 1;
  v4 = v73;
  v5 = *(_QWORD *)(a2 + 80);
  v70 = v73;
  v6 = v5 - 24;
  v72 = 16;
  if ( !v5 )
    v6 = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v65 = v6;
  v73[0] = v6;
  v73[1] = v6;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v71 = 1;
  do
  {
    v7 = v2;
    v8 = v2 - 1;
    v9 = &v4[2 * v7 - 2];
    v10 = *v9;
    v11 = v9[1];
    v71 = v8;
    v12 = *(unsigned __int8 *)(sub_157ED20(v10, v5, v8) + 16) - 34;
    if ( v12 <= 0x36 && ((1LL << v12) & 0x40018000000001LL) != 0 )
      v11 = v10;
    v13 = *(_DWORD *)(a1 + 24);
    if ( !v13 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_74;
    }
    v14 = *(_QWORD *)(a1 + 8);
    v15 = (v13 - 1) & (((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v10 != *v16 )
    {
      v47 = 1;
      v48 = 0;
      while ( v17 != -8 )
      {
        if ( !v48 && v17 == -16 )
          v48 = v16;
        v15 = (v13 - 1) & (v47 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v10 == *v16 )
          goto LABEL_9;
        ++v47;
      }
      v49 = *(_DWORD *)(a1 + 16);
      if ( v48 )
        v16 = v48;
      ++*(_QWORD *)a1;
      v50 = v49 + 1;
      if ( 4 * v50 < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(a1 + 20) - v50 > v13 >> 3 )
          goto LABEL_69;
        v68 = ((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9);
        sub_14DDDA0(a1, v13);
        v58 = *(_DWORD *)(a1 + 24);
        if ( !v58 )
LABEL_104:
          JUMPOUT(0x419798);
        v59 = v58 - 1;
        v60 = *(_QWORD *)(a1 + 8);
        v57 = 0;
        v61 = 1;
        v62 = v59 & v68;
        v50 = *(_DWORD *)(a1 + 16) + 1;
        v16 = (__int64 *)(v60 + 16LL * (v59 & v68));
        v63 = *v16;
        if ( v10 == *v16 )
          goto LABEL_69;
        while ( v63 != -8 )
        {
          if ( !v57 && v63 == -16 )
            v57 = v16;
          v62 = v59 & (v61 + v62);
          v16 = (__int64 *)(v60 + 16LL * v62);
          v63 = *v16;
          if ( v10 == *v16 )
            goto LABEL_69;
          ++v61;
        }
        goto LABEL_86;
      }
LABEL_74:
      sub_14DDDA0(a1, 2 * v13);
      v51 = *(_DWORD *)(a1 + 24);
      if ( !v51 )
        goto LABEL_104;
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 8);
      v54 = (v51 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v50 = *(_DWORD *)(a1 + 16) + 1;
      v16 = (__int64 *)(v53 + 16LL * v54);
      v55 = *v16;
      if ( v10 == *v16 )
        goto LABEL_69;
      v56 = 1;
      v57 = 0;
      while ( v55 != -8 )
      {
        if ( !v57 && v55 == -16 )
          v57 = v16;
        v54 = v52 & (v56 + v54);
        v16 = (__int64 *)(v53 + 16LL * v54);
        v55 = *v16;
        if ( v10 == *v16 )
          goto LABEL_69;
        ++v56;
      }
LABEL_86:
      if ( v57 )
        v16 = v57;
LABEL_69:
      *(_DWORD *)(a1 + 16) = v50;
      if ( *v16 != -8 )
        --*(_DWORD *)(a1 + 20);
      *v16 = v10;
      v22 = v16 + 1;
      v16[1] = 0;
      v18 = (__int64)(v16 + 1);
      goto LABEL_72;
    }
LABEL_9:
    v18 = (__int64)(v16 + 1);
    v19 = v16[1];
    v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
    v21 = (v19 >> 2) & 1;
    if ( !v21 )
    {
      if ( v20 )
      {
        v24 = v16 + 2;
        v22 = v16 + 1;
        goto LABEL_50;
      }
      v22 = v16 + 1;
LABEL_72:
      v20 = 0;
      v5 = 0;
      goto LABEL_18;
    }
    v22 = *(_QWORD **)v20;
    v23 = 8LL * *(unsigned int *)(v20 + 8);
    v24 = (_QWORD *)(*(_QWORD *)v20 + v23);
    v25 = v23 >> 3;
    v26 = v23 >> 5;
    if ( v26 )
    {
      while ( *v22 != v11 )
      {
        if ( v22[1] == v11 )
        {
          ++v22;
          goto LABEL_17;
        }
        if ( v22[2] == v11 )
        {
          v22 += 2;
          goto LABEL_17;
        }
        if ( v22[3] == v11 )
        {
          v22 += 3;
          goto LABEL_17;
        }
        v22 += 4;
        if ( !--v26 )
        {
          v25 = v24 - v22;
          goto LABEL_47;
        }
      }
      goto LABEL_17;
    }
LABEL_47:
    if ( v25 != 2 )
    {
      if ( v25 == 3 )
      {
        if ( *v22 == v11 )
          goto LABEL_59;
        if ( *++v22 == v11 )
          goto LABEL_59;
        goto LABEL_49;
      }
      if ( v25 == 1 )
        goto LABEL_50;
      v22 = v24;
LABEL_59:
      v24 = v22;
LABEL_60:
      v22 = v24;
LABEL_17:
      v5 = 1;
      v18 = *(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8);
      goto LABEL_18;
    }
    if ( *v22 == v11 )
      goto LABEL_59;
LABEL_49:
    ++v22;
LABEL_50:
    if ( *v22 == v11 )
      v24 = v22;
    if ( v21 )
      goto LABEL_60;
    v22 = v24;
    if ( v20 )
      v18 = (__int64)(v16 + 2);
    v5 = 0;
LABEL_18:
    v2 = v71;
    if ( v22 == (_QWORD *)v18 )
    {
      if ( v20 )
      {
        v66 = v20;
        if ( !(_BYTE)v5 )
        {
          v28 = sub_22077B0(48);
          v29 = v66;
          if ( v28 )
          {
            *(_QWORD *)v28 = v28 + 16;
            *(_QWORD *)(v28 + 8) = 0x400000000LL;
          }
          v30 = v28;
          v31 = v28 & 0xFFFFFFFFFFFFFFF8LL;
          v16[1] = v30 | 4;
          v32 = *(unsigned int *)(v31 + 8);
          if ( (unsigned int)v32 >= *(_DWORD *)(v31 + 12) )
          {
            v5 = v31 + 16;
            v64 = v66;
            v69 = v31;
            sub_16CD150(v31, v31 + 16, 0, 8);
            v31 = v69;
            v29 = v64;
            v32 = *(unsigned int *)(v69 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v31 + 8 * v32) = v29;
          ++*(_DWORD *)(v31 + 8);
        }
        v33 = v16[1] & 0xFFFFFFFFFFFFFFF8LL;
        v34 = *(unsigned int *)(v33 + 8);
        if ( (unsigned int)v34 >= *(_DWORD *)(v33 + 12) )
        {
          v5 = v33 + 16;
          sub_16CD150(v33, v33 + 16, 0, 8);
          v34 = *(unsigned int *)(v33 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v33 + 8 * v34) = v11;
        ++*(_DWORD *)(v33 + 8);
      }
      else
      {
        v16[1] = v11;
      }
      v35 = sub_157EBA0(v10);
      v36 = v35;
      if ( *(_BYTE *)(v35 + 16) == 33 )
      {
        v37 = *(_QWORD *)(*(_QWORD *)(v35 - 48) - 24LL);
        v38 = (*(_BYTE *)(v37 + 23) & 0x40) != 0
            ? *(__int64 **)(v37 - 8)
            : (__int64 *)(v37 - 24LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF));
        v39 = *v38;
        v11 = v65;
        if ( *(_BYTE *)(v39 + 16) != 16 )
          v11 = *(_QWORD *)(v39 + 40);
      }
      v40 = 0;
      v41 = sub_15F4D60(v36);
      v42 = sub_157EBA0(v10);
      v2 = v71;
      v43 = v42;
      if ( v41 )
      {
        do
        {
          v5 = v40;
          v45 = sub_15F4DF0(v43, v40);
          v46 = v71;
          if ( v71 >= v72 )
          {
            v5 = (__int64)v73;
            v67 = v45;
            sub_16CD150(&v70, v73, 0, 16);
            v46 = v71;
            v45 = v67;
          }
          v44 = &v70[2 * v46];
          ++v40;
          *v44 = v45;
          v44[1] = v11;
          v2 = ++v71;
        }
        while ( v41 != v40 );
      }
    }
    v4 = v70;
  }
  while ( v2 );
  if ( v70 != v73 )
    _libc_free((unsigned __int64)v70);
  return a1;
}
