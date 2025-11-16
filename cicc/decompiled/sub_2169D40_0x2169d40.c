// Function: sub_2169D40
// Address: 0x2169d40
//
__int64 __fastcall sub_2169D40(__int64 *a1, int a2, __int64 a3, unsigned int a4, unsigned int *a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // rbx
  unsigned int v9; // r14d
  __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned int v12; // r13d
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rcx
  unsigned __int64 v18; // r8
  unsigned int v19; // esi
  unsigned int v20; // eax
  unsigned __int8 v21; // al
  char v22; // al
  int v23; // r13d
  __int64 *v24; // r14
  int v25; // ebx
  char v26; // al
  __int64 v27; // rdx
  int v28; // eax
  int v29; // eax
  __int64 v30; // rcx
  unsigned int v31; // esi
  unsigned __int64 v32; // rcx
  __int64 v33; // rax
  size_t v34; // rdx
  void *v35; // r8
  unsigned int *v36; // r10
  unsigned int v37; // esi
  unsigned int v38; // edi
  unsigned int v39; // eax
  __int64 v40; // r12
  _QWORD *v41; // r15
  unsigned int v42; // ebx
  __int64 v43; // rdi
  unsigned int v44; // eax
  __int64 v45; // rcx
  unsigned int v46; // r15d
  __int64 v47; // rdx
  unsigned int v48; // r15d
  int v49; // r12d
  __int64 v50; // rdx
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // r9
  unsigned int v55; // r15d
  __int64 *v56; // r13
  unsigned int v57; // ebx
  int v58; // r14d
  __int64 v59; // rdx
  int v60; // r14d
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // [rsp+20h] [rbp-80h]
  __int64 v66; // [rsp+20h] [rbp-80h]
  unsigned int v67; // [rsp+28h] [rbp-78h]
  char v68; // [rsp+28h] [rbp-78h]
  unsigned int v69; // [rsp+30h] [rbp-70h]
  unsigned int v70; // [rsp+30h] [rbp-70h]
  __int64 *v71; // [rsp+30h] [rbp-70h]
  __int64 v72; // [rsp+30h] [rbp-70h]
  _QWORD *v73; // [rsp+38h] [rbp-68h]
  __int64 v76; // [rsp+50h] [rbp-50h]
  unsigned __int64 v77; // [rsp+58h] [rbp-48h]
  unsigned int *v78; // [rsp+58h] [rbp-48h]
  char v79; // [rsp+58h] [rbp-48h]
  void *v80; // [rsp+58h] [rbp-48h]
  size_t n; // [rsp+68h] [rbp-38h]
  int na; // [rsp+68h] [rbp-38h]
  size_t nb; // [rsp+68h] [rbp-38h]
  size_t nc; // [rsp+68h] [rbp-38h]
  char ne; // [rsp+68h] [rbp-38h]
  char nf; // [rsp+68h] [rbp-38h]
  char ng; // [rsp+68h] [rbp-38h]
  unsigned int nd; // [rsp+68h] [rbp-38h]
  size_t nh; // [rsp+68h] [rbp-38h]

  if ( *(_BYTE *)(a3 + 8) != 16 )
    BUG();
  v6 = a3;
  v7 = a1;
  v76 = *(_QWORD *)(a3 + 32);
  v9 = (unsigned int)v76 / a4;
  v73 = sub_16463B0(*(__int64 **)(a3 + 24), (unsigned int)v76 / a4);
  v11 = sub_1F43D80(v7[2], *v7, v6, v10);
  v12 = v11;
  n = HIDWORD(v11);
  if ( *(_BYTE *)(v6 + 8) != 16 || (v77 = v11, v19 = sub_1643030(v6), v20 = sub_2165D90(n), v11 = v77, v19 >= v20) )
  {
    v13 = a1[2];
    goto LABEL_4;
  }
  v21 = sub_2167220(*a1, v6);
  v13 = a1[2];
  v11 = v77;
  if ( a2 == 31 )
  {
    if ( v21 && (_BYTE)n )
    {
      v22 = *(_BYTE *)(v21 + v13 + 115LL * (unsigned __int8)n + 58658);
LABEL_11:
      if ( (v22 & 0xFB) == 0 )
        goto LABEL_4;
    }
  }
  else if ( v21 && (_BYTE)n )
  {
    v22 = (unsigned __int8)*(_WORD *)(v13 + 2 * (v21 + 115LL * (unsigned __int8)n + 16104)) >> 4;
    goto LABEL_11;
  }
  na = *(_QWORD *)(v6 + 32);
  if ( na > 0 )
  {
    v23 = 0;
    v69 = v9;
    v24 = v7;
    v25 = 0;
    while ( 1 )
    {
      v26 = *(_BYTE *)(v6 + 8);
      if ( a2 == 31 )
      {
        v27 = v6;
        if ( v26 == 16 )
          goto LABEL_76;
      }
      else
      {
        v27 = v6;
        if ( v26 == 16 )
LABEL_76:
          v27 = **(_QWORD **)(v6 + 16);
      }
      ++v23;
      v28 = sub_1F43D80(v13, *v24, v27, v11);
      v13 = v24[2];
      v25 += v28;
      if ( na == v23 )
      {
        v11 = v77;
        v29 = v25;
        v7 = v24;
        v9 = v69;
        v12 = v77 + v29;
        break;
      }
    }
  }
LABEL_4:
  v14 = sub_1F43D80(v13, *v7, v6, v11);
  v15 = *v7;
  v16 = v6;
  v17 = 1;
  v18 = HIDWORD(v14);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v16 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v52 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 24);
        v17 *= v52;
        continue;
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 9:
      case 0xB:
        goto LABEL_18;
      case 7:
        ng = v18;
        sub_15A9520(v15, 0);
        LOBYTE(v18) = ng;
        break;
      case 0xD:
        ne = v18;
        sub_15A9930(v15, v16);
        LOBYTE(v18) = ne;
        break;
      case 0xE:
        v68 = v18;
        v72 = *(_QWORD *)(v16 + 24);
        sub_15A9FE0(v15, v72);
        v53 = v72;
        v54 = 1;
        LOBYTE(v18) = v68;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v53 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v63 = *(_QWORD *)(v53 + 32);
              v53 = *(_QWORD *)(v53 + 24);
              v54 *= v63;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_18;
            case 7:
              v62 = 0;
              v79 = v68;
              goto LABEL_69;
            case 0xD:
              sub_15A9930(v15, v53);
              LOBYTE(v18) = v68;
              goto LABEL_18;
            case 0xE:
              v66 = *(_QWORD *)(v53 + 24);
              sub_15A9FE0(v15, v66);
              sub_127FA20(v15, v66);
              JUMPOUT(0x216A3FF);
            case 0xF:
              v62 = *(_DWORD *)(v53 + 8) >> 8;
              v79 = v68;
LABEL_69:
              sub_15A9520(v15, v62);
              LOBYTE(v18) = v79;
              break;
          }
          break;
        }
        break;
      case 0xF:
        nf = v18;
        sub_15A9520(v15, *(_DWORD *)(v16 + 8) >> 8);
        LOBYTE(v18) = nf;
        break;
    }
    break;
  }
LABEL_18:
  v31 = ((unsigned int)sub_2165D90(v18) + 7) >> 3;
  v32 = (unsigned __int64)(v30 + 7) >> 3;
  if ( a2 == 30 )
  {
    if ( v31 >= (unsigned int)v32 )
    {
      v78 = &a5[a6];
      goto LABEL_32;
    }
    v67 = (v31 + (unsigned int)v32 - 1) / v31;
    v70 = (v67 + 63) >> 6;
    nb = 8LL * v70;
    v33 = malloc(nb);
    v34 = nb;
    v35 = (void *)v33;
    if ( !v33 )
    {
      if ( nb || (v64 = malloc(1u), v34 = 0, v35 = 0, !v64) )
      {
        v80 = v35;
        nh = v34;
        sub_16BD1C0("Allocation failed", 1u);
        v34 = nh;
        v35 = v80;
      }
      else
      {
        v35 = (void *)v64;
      }
    }
    v78 = &a5[a6];
    if ( v70 )
    {
      v35 = memset(v35, 0, v34);
      if ( a5 == v78 )
      {
LABEL_28:
        v65 = v6;
        v40 = 0;
        v41 = v35;
        v71 = v7;
        v42 = 0;
        do
        {
          v43 = v41[v40++];
          v42 += sub_39FAC40(v43);
        }
        while ( (v67 + 63) >> 6 > (unsigned int)v40 );
        v44 = v42;
        v7 = v71;
        v6 = v65;
        v35 = v41;
        v12 *= v44 / v67;
        goto LABEL_31;
      }
    }
    else if ( a5 == v78 )
    {
      goto LABEL_80;
    }
    v36 = a5;
    do
    {
      v37 = *v36;
      v38 = 0;
      if ( (unsigned int)v76 >= a4 )
      {
        do
        {
          ++v38;
          v39 = v37 / ((v67 + (unsigned int)v76 - 1) / v67);
          v37 += a4;
          *((_QWORD *)v35 + (v39 >> 6)) |= 1LL << v39;
        }
        while ( v9 > v38 );
      }
      ++v36;
    }
    while ( v78 != v36 );
    if ( v70 )
      goto LABEL_28;
LABEL_80:
    v12 = 0;
LABEL_31:
    _libc_free((unsigned __int64)v35);
LABEL_32:
    v45 = (__int64)v78;
    if ( a5 != v78 )
    {
      nc = (size_t)a5;
      do
      {
        v45 = a4;
        if ( (unsigned int)v76 >= a4 )
        {
          v46 = 0;
          do
          {
            v47 = v6;
            if ( *(_BYTE *)(v6 + 8) == 16 )
              v47 = **(_QWORD **)(v6 + 16);
            ++v46;
            v12 += sub_1F43D80(v7[2], *v7, v47, v45);
          }
          while ( v9 > v46 );
        }
        nc += 4LL;
      }
      while ( v78 != (unsigned int *)nc );
    }
    if ( (unsigned int)v76 >= a4 )
    {
      v48 = 0;
      v49 = 0;
      do
      {
        v50 = (__int64)v73;
        if ( *((_BYTE *)v73 + 8) == 16 )
          v50 = *(_QWORD *)v73[2];
        ++v48;
        v49 += sub_1F43D80(v7[2], *v7, v50, v45);
      }
      while ( v9 > v48 );
      v12 += a6 * v49;
    }
    return v12;
  }
  if ( (unsigned int)v76 >= a4 )
  {
    v55 = 0;
    nd = v12;
    v56 = v7;
    v57 = v9;
    v58 = 0;
    do
    {
      v59 = (__int64)v73;
      if ( *((_BYTE *)v73 + 8) == 16 )
        v59 = *(_QWORD *)v73[2];
      ++v55;
      v58 += sub_1F43D80(v56[2], *v56, v59, v32);
    }
    while ( v57 > v55 );
    v7 = v56;
    v12 = v58 * a4 + nd;
  }
  if ( (_DWORD)v76 )
  {
    v60 = 0;
    do
    {
      v61 = v6;
      if ( *(_BYTE *)(v6 + 8) == 16 )
        v61 = **(_QWORD **)(v6 + 16);
      ++v60;
      v12 += sub_1F43D80(v7[2], *v7, v61, v32);
    }
    while ( (_DWORD)v76 != v60 );
  }
  return v12;
}
