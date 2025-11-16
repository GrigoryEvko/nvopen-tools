// Function: sub_1BC8120
// Address: 0x1bc8120
//
_DWORD *__fastcall sub_1BC8120(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _DWORD *a5,
        __int64 a6,
        const void *a7,
        __int64 a8,
        __m128i a9)
{
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // r9
  unsigned __int64 v17; // rax
  __int64 *v18; // r14
  unsigned __int64 v19; // r10
  signed __int64 v20; // rdx
  char *v21; // r8
  __int64 v22; // rcx
  char *v23; // r11
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 *v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // r8
  __m128i v31; // xmm0
  __int64 v32; // r14
  int v33; // esi
  unsigned int v34; // edx
  unsigned __int64 v35; // rax
  char v36; // cl
  unsigned int v37; // esi
  unsigned int v38; // edx
  int v39; // edi
  __int64 v40; // rdx
  int v42; // r11d
  unsigned __int64 v43; // r10
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // rdi
  unsigned int v47; // r10d
  __int64 *v48; // rax
  __int64 *v49; // rcx
  int v50; // ecx
  int v51; // ecx
  unsigned int v52; // edx
  __int64 v53; // rdi
  int v54; // ecx
  __int64 v55; // rdi
  int v56; // ecx
  unsigned int v57; // edx
  int v58; // r11d
  void *v59; // rdi
  size_t v60; // r10
  size_t v61; // rax
  int v62; // r11d
  size_t v63; // [rsp+0h] [rbp-90h]
  size_t v64; // [rsp+8h] [rbp-88h]
  size_t v65; // [rsp+10h] [rbp-80h]
  char *v66; // [rsp+10h] [rbp-80h]
  __int64 v67; // [rsp+18h] [rbp-78h]
  char *v68; // [rsp+18h] [rbp-78h]
  char *v69; // [rsp+18h] [rbp-78h]
  char *v70; // [rsp+20h] [rbp-70h]
  __int64 v71; // [rsp+20h] [rbp-70h]
  __int64 v72; // [rsp+20h] [rbp-70h]
  signed __int64 v73; // [rsp+28h] [rbp-68h]
  __int64 v74; // [rsp+28h] [rbp-68h]
  __int64 v75; // [rsp+28h] [rbp-68h]
  char *v76; // [rsp+30h] [rbp-60h]
  int v77; // [rsp+30h] [rbp-60h]
  char *v78; // [rsp+30h] [rbp-60h]
  char *v79; // [rsp+30h] [rbp-60h]
  int v80; // [rsp+40h] [rbp-50h]
  __int64 v81; // [rsp+40h] [rbp-50h]
  int v82; // [rsp+48h] [rbp-48h]
  char v83; // [rsp+4Ch] [rbp-44h]

  v9 = a2;
  v10 = a2;
  v12 = *(_QWORD *)(a1 + 8);
  v83 = a4;
  if ( v12 == *(_QWORD *)(a1 + 16) )
  {
    sub_1BC0220((unsigned __int64 **)a1, (unsigned __int64 *)v12, a1, a4, (int)a5);
    v13 = *(_QWORD *)(a1 + 8);
  }
  else
  {
    if ( v12 )
    {
      *(_BYTE *)(v12 + 88) = 0;
      *(_QWORD *)v12 = v12 + 16;
      *(_QWORD *)(v12 + 8) = 0x800000000LL;
      *(_QWORD *)(v12 + 96) = v12 + 112;
      *(_QWORD *)(v12 + 104) = 0x400000000LL;
      *(_QWORD *)(v12 + 152) = v12 + 168;
      *(_QWORD *)(v12 + 80) = 0;
      *(_QWORD *)(v12 + 128) = 0;
      *(_QWORD *)(v12 + 136) = 0;
      *(_QWORD *)(v12 + 144) = a1;
      *(_QWORD *)(v12 + 160) = 0x100000000LL;
      v12 = *(_QWORD *)(a1 + 8);
    }
    v13 = v12 + 176;
    *(_QWORD *)(a1 + 8) = v13;
  }
  v82 = -1171354717 * ((v13 - *(_QWORD *)a1) >> 4) - 1;
  v14 = *(_QWORD *)a1 + 176LL * v82;
  v15 = *(unsigned int *)(v14 + 8);
  v16 = 8 * a3;
  v17 = *(unsigned int *)(v14 + 12);
  v18 = &v10[a3];
  v19 = (8 * a3) >> 3;
  v20 = 8 * v15;
  if ( 8 * v15 )
  {
    if ( v15 + v19 > v17 )
    {
      sub_16CD150(*(_QWORD *)a1 + 176LL * v82, (const void *)(v14 + 16), v15 + v19, 8, v15 + v19, v16);
      v15 = *(unsigned int *)(v14 + 8);
      v19 = (8 * a3) >> 3;
      v16 = 8 * a3;
      v20 = 8 * v15;
    }
    v21 = *(char **)v14;
    v22 = v20 >> 3;
    v23 = (char *)(*(_QWORD *)v14 + v20);
    if ( v16 <= (unsigned __int64)v20 )
    {
      v59 = (void *)(*(_QWORD *)v14 + v20);
      v60 = v20 - v16;
      v78 = &v21[v20 - v16];
      v61 = v16;
      v74 = v16 >> 3;
      if ( v16 >> 3 > (unsigned __int64)*(unsigned int *)(v14 + 12) - v15 )
      {
        v63 = v16;
        v64 = v20 - v16;
        v66 = (char *)(*(_QWORD *)v14 + v20);
        v69 = *(char **)v14;
        v72 = v16;
        sub_16CD150(v14, (const void *)(v14 + 16), (v16 >> 3) + v15, 8, (int)v21, v16);
        v15 = *(unsigned int *)(v14 + 8);
        v61 = v63;
        v60 = v64;
        v23 = v66;
        v21 = v69;
        v59 = (void *)(*(_QWORD *)v14 + 8 * v15);
        v16 = v72;
      }
      if ( v23 != v78 )
      {
        v65 = v60;
        v68 = v21;
        v71 = v16;
        memmove(v59, v78, v61);
        LODWORD(v15) = *(_DWORD *)(v14 + 8);
        v60 = v65;
        v21 = v68;
        v16 = v71;
      }
      *(_DWORD *)(v14 + 8) = v74 + v15;
      if ( v21 != v78 )
      {
        v75 = v16;
        v79 = v21;
        memmove(&v21[v16], v21, v60);
        v16 = v75;
        v21 = v79;
      }
      if ( v18 != v10 )
        memmove(v21, v10, v16);
    }
    else
    {
      v24 = v19 + v15;
      *(_DWORD *)(v14 + 8) = v24;
      if ( v21 != v23 )
      {
        v67 = v20 >> 3;
        v70 = v23;
        v73 = v20;
        v76 = v21;
        memcpy(&v21[8 * v24 - v20], v21, v20);
        v22 = v67;
        v23 = v70;
        v20 = v73;
        v21 = v76;
      }
      if ( v20 )
      {
        v25 = 0;
        do
        {
          *(_QWORD *)&v21[8 * v25] = v10[v25];
          ++v25;
        }
        while ( v22 != v25 );
        v26 = (__int64 *)((char *)v10 + v20);
      }
      else
      {
        v26 = v10;
      }
      if ( v18 != v26 )
        memcpy(v23, v26, (char *)v18 - (char *)v26);
    }
  }
  else
  {
    if ( v17 - v15 < v19 )
    {
      sub_16CD150(*(_QWORD *)a1 + 176LL * v82, (const void *)(v14 + 16), v15 + v19, 8, (int)a5, v16);
      v15 = *(unsigned int *)(v14 + 8);
      v16 = 8 * a3;
      v19 = (8 * a3) >> 3;
      v20 = 8 * v15;
    }
    if ( v18 != v10 )
    {
      v77 = v19;
      memcpy((void *)(v20 + *(_QWORD *)v14), v10, v16);
      LODWORD(v15) = *(_DWORD *)(v14 + 8);
      LODWORD(v19) = v77;
    }
    *(_DWORD *)(v14 + 8) = v19 + v15;
  }
  v27 = *(unsigned int *)(v14 + 108);
  v28 = 4 * a8;
  *(_BYTE *)(v14 + 88) = v83 ^ 1;
  v29 = *(unsigned int *)(v14 + 104);
  v30 = (4 * a8) >> 2;
  if ( v30 > v27 - v29 )
  {
    v81 = v28 >> 2;
    sub_16CD150(v14 + 96, (const void *)(v14 + 112), v30 + v29, 4, v30, v28);
    v29 = *(unsigned int *)(v14 + 104);
    v28 = 4 * a8;
    LODWORD(v30) = v81;
  }
  if ( v28 )
  {
    v80 = v30;
    memcpy((void *)(*(_QWORD *)(v14 + 96) + 4 * v29), a7, v28);
    LODWORD(v29) = *(_DWORD *)(v14 + 104);
    LODWORD(v30) = v80;
  }
  v31 = _mm_loadu_si128(&a9);
  *(_DWORD *)(v14 + 104) = v30 + v29;
  *(__m128i *)(v14 + 128) = v31;
  if ( !v83 )
  {
    if ( v18 == v10 )
      goto LABEL_37;
    v30 = *(_QWORD *)(a1 + 120);
    v28 = *(_QWORD *)(a1 + 112);
    while ( 1 )
    {
      v45 = *v9;
      if ( v30 != v28 )
        goto LABEL_50;
      v46 = (__int64 *)(v30 + 8LL * *(unsigned int *)(a1 + 132));
      v47 = *(_DWORD *)(a1 + 132);
      if ( v46 != (__int64 *)v30 )
      {
        v48 = (__int64 *)v30;
        v49 = 0;
        while ( v45 != *v48 )
        {
          if ( *v48 == -2 )
            v49 = v48;
          if ( v46 == ++v48 )
          {
            if ( !v49 )
              goto LABEL_76;
            *v49 = v45;
            v30 = *(_QWORD *)(a1 + 120);
            --*(_DWORD *)(a1 + 136);
            v28 = *(_QWORD *)(a1 + 112);
            ++*(_QWORD *)(a1 + 104);
            goto LABEL_51;
          }
        }
        goto LABEL_51;
      }
LABEL_76:
      if ( v47 < *(_DWORD *)(a1 + 128) )
      {
        *(_DWORD *)(a1 + 132) = v47 + 1;
        *v46 = v45;
        v28 = *(_QWORD *)(a1 + 112);
        ++*(_QWORD *)(a1 + 104);
        v30 = *(_QWORD *)(a1 + 120);
      }
      else
      {
LABEL_50:
        sub_16CCBA0(a1 + 104, v45);
        v30 = *(_QWORD *)(a1 + 120);
        v28 = *(_QWORD *)(a1 + 112);
      }
LABEL_51:
      if ( v18 == ++v9 )
        goto LABEL_37;
    }
  }
  if ( (_DWORD)a3 )
  {
    v32 = (__int64)&v10[(unsigned int)(a3 - 1) + 1];
    while ( 1 )
    {
      v36 = *(_BYTE *)(a1 + 32) & 1;
      if ( v36 )
      {
        v30 = a1 + 40;
        v33 = 3;
      }
      else
      {
        v37 = *(_DWORD *)(a1 + 48);
        v30 = *(_QWORD *)(a1 + 40);
        if ( !v37 )
        {
          v38 = *(_DWORD *)(a1 + 32);
          ++*(_QWORD *)(a1 + 24);
          v35 = 0;
          v39 = (v38 >> 1) + 1;
LABEL_31:
          LODWORD(v30) = 3 * v37;
          goto LABEL_32;
        }
        v33 = v37 - 1;
      }
      v34 = v33 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v35 = v30 + 16LL * v34;
      v28 = *(_QWORD *)v35;
      if ( *v10 == *(_QWORD *)v35 )
      {
LABEL_26:
        ++v10;
        *(_DWORD *)(v35 + 8) = v82;
        if ( (__int64 *)v32 == v10 )
          break;
      }
      else
      {
        v42 = 1;
        v43 = 0;
        while ( v28 != -8 )
        {
          if ( v28 == -16 && !v43 )
            v43 = v35;
          v34 = v33 & (v42 + v34);
          v35 = v30 + 16LL * v34;
          v28 = *(_QWORD *)v35;
          if ( *v10 == *(_QWORD *)v35 )
            goto LABEL_26;
          ++v42;
        }
        v38 = *(_DWORD *)(a1 + 32);
        LODWORD(v30) = 12;
        v37 = 4;
        if ( v43 )
          v35 = v43;
        ++*(_QWORD *)(a1 + 24);
        v39 = (v38 >> 1) + 1;
        if ( !v36 )
        {
          v37 = *(_DWORD *)(a1 + 48);
          goto LABEL_31;
        }
LABEL_32:
        if ( 4 * v39 >= (unsigned int)v30 )
        {
          sub_1BC7D50(a1 + 24, 2 * v37);
          if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
          {
            v30 = a1 + 40;
            v51 = 3;
          }
          else
          {
            v50 = *(_DWORD *)(a1 + 48);
            v30 = *(_QWORD *)(a1 + 40);
            if ( !v50 )
              goto LABEL_110;
            v51 = v50 - 1;
          }
          v52 = v51 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
          v35 = v30 + 16LL * v52;
          v53 = *(_QWORD *)v35;
          if ( *v10 != *(_QWORD *)v35 )
          {
            v62 = 1;
            v28 = 0;
            while ( v53 != -8 )
            {
              if ( !v28 && v53 == -16 )
                v28 = v35;
              v52 = v51 & (v62 + v52);
              v35 = v30 + 16LL * v52;
              v53 = *(_QWORD *)v35;
              if ( *v10 == *(_QWORD *)v35 )
                goto LABEL_65;
              ++v62;
            }
LABEL_72:
            if ( v28 )
              v35 = v28;
          }
LABEL_65:
          v38 = *(_DWORD *)(a1 + 32);
          goto LABEL_34;
        }
        if ( v37 - *(_DWORD *)(a1 + 36) - v39 <= v37 >> 3 )
        {
          sub_1BC7D50(a1 + 24, v37);
          if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
          {
            v55 = a1 + 40;
            v56 = 3;
          }
          else
          {
            v54 = *(_DWORD *)(a1 + 48);
            v55 = *(_QWORD *)(a1 + 40);
            if ( !v54 )
            {
LABEL_110:
              *(_DWORD *)(a1 + 32) = (2 * (*(_DWORD *)(a1 + 32) >> 1) + 2) | *(_DWORD *)(a1 + 32) & 1;
              BUG();
            }
            v56 = v54 - 1;
          }
          v57 = v56 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
          v35 = v55 + 16LL * v57;
          v30 = *(_QWORD *)v35;
          if ( *v10 != *(_QWORD *)v35 )
          {
            v58 = 1;
            v28 = 0;
            while ( v30 != -8 )
            {
              if ( !v28 && v30 == -16 )
                v28 = v35;
              v57 = v56 & (v58 + v57);
              v35 = v55 + 16LL * v57;
              v30 = *(_QWORD *)v35;
              if ( *v10 == *(_QWORD *)v35 )
                goto LABEL_65;
              ++v58;
            }
            goto LABEL_72;
          }
          goto LABEL_65;
        }
LABEL_34:
        *(_DWORD *)(a1 + 32) = (2 * (v38 >> 1) + 2) | v38 & 1;
        if ( *(_QWORD *)v35 != -8 )
          --*(_DWORD *)(a1 + 36);
        v40 = *v10++;
        *(_DWORD *)(v35 + 8) = 0;
        *(_DWORD *)(v35 + 8) = v82;
        *(_QWORD *)v35 = v40;
        if ( (__int64 *)v32 == v10 )
          break;
      }
    }
  }
LABEL_37:
  if ( (int)*a5 >= 0 )
  {
    v44 = *(unsigned int *)(v14 + 160);
    if ( (unsigned int)v44 >= *(_DWORD *)(v14 + 164) )
    {
      sub_16CD150(v14 + 152, (const void *)(v14 + 168), 0, 4, v30, v28);
      v44 = *(unsigned int *)(v14 + 160);
    }
    *(_DWORD *)(*(_QWORD *)(v14 + 152) + 4 * v44) = *a5;
    ++*(_DWORD *)(v14 + 160);
  }
  *a5 = v82;
  return a5;
}
