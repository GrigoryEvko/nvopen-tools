// Function: sub_37E2870
// Address: 0x37e2870
//
_DWORD *__fastcall sub_37E2870(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r15
  int v5; // ebx
  __int64 v6; // r14
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _DWORD *v11; // rax
  __int64 v12; // rdx
  _DWORD *i; // rdx
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // ecx
  __int64 v17; // rdx
  _DWORD *v18; // rax
  __int64 v19; // rdx
  _DWORD *j; // rdx
  __int64 v21; // r8
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // r15
  unsigned __int64 v27; // rbx
  unsigned __int64 **v28; // rax
  unsigned __int64 *v29; // r14
  unsigned __int64 **v30; // rax
  unsigned __int64 *v31; // r13
  __int64 v32; // r12
  unsigned int v33; // eax
  unsigned int v34; // eax
  unsigned int v35; // ecx
  __int64 v36; // rdx
  _DWORD *v37; // rax
  __int64 v38; // rdx
  _DWORD *k; // rdx
  unsigned int v40; // eax
  _DWORD *result; // rax
  unsigned int v42; // ecx
  __int64 v43; // rdx
  __int64 v44; // rdx
  _DWORD *m; // rdx
  unsigned int v46; // eax
  unsigned int v47; // ebx
  __int64 v48; // rdi
  char v49; // al
  __int64 v50; // rax
  unsigned int v51; // eax
  int v52; // ecx
  char v53; // al
  __int64 v54; // rdi
  int v55; // ecx
  unsigned int v56; // eax
  int v57; // ecx
  __int64 v58; // rdi
  int v59; // ecx
  char v60; // al
  unsigned int v61; // eax
  unsigned int v62; // ebx
  char v63; // al
  __int64 v64; // rdi
  __int64 v65; // rax
  int v66; // [rsp+Ch] [rbp-44h]
  int v67; // [rsp+Ch] [rbp-44h]
  unsigned int v68; // [rsp+Ch] [rbp-44h]
  unsigned int v69; // [rsp+Ch] [rbp-44h]
  __int64 v70; // [rsp+18h] [rbp-38h]

  v4 = *(unsigned int *)(a2 + 24);
  v5 = *(_DWORD *)(a2 + 24);
  v70 = 856 * v4;
  v6 = **(_QWORD **)a1 + 856 * v4;
  v7 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)(v6 + 8);
  v8 = v7 >> 1;
  if ( v8 )
  {
    if ( (*(_BYTE *)(v6 + 16) & 1) == 0 )
    {
      v9 = 4 * v8;
      goto LABEL_4;
    }
LABEL_65:
    v11 = (_DWORD *)(v6 + 24);
    v12 = 16;
    goto LABEL_7;
  }
  if ( !*(_DWORD *)(v6 + 20) )
    goto LABEL_10;
  v9 = 0;
  if ( (*(_BYTE *)(v6 + 16) & 1) != 0 )
    goto LABEL_65;
LABEL_4:
  v10 = *(unsigned int *)(v6 + 32);
  if ( (unsigned int)v10 > v9 && (unsigned int)v10 > 0x40 )
  {
    if ( v8 && (v51 = v8 - 1) != 0 )
    {
      _BitScanReverse(&v51, v51);
      v52 = 1 << (33 - (v51 ^ 0x1F));
      if ( (unsigned int)(v52 - 9) > 0x36 )
      {
        if ( (_DWORD)v10 == v52 )
          goto LABEL_106;
        v69 = 1 << (33 - (v51 ^ 0x1F));
        sub_C7D6A0(*(_QWORD *)(v6 + 24), 8 * v10, 4);
        v55 = v69;
        v53 = *(_BYTE *)(v6 + 16) | 1;
        *(_BYTE *)(v6 + 16) = v53;
        if ( v69 <= 8 )
          goto LABEL_106;
        v54 = 8LL * v69;
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(v6 + 24), 8 * v10, 4);
        v53 = *(_BYTE *)(v6 + 16);
        v54 = 512;
        v55 = 64;
      }
      v66 = v55;
      *(_BYTE *)(v6 + 16) = v53 & 0xFE;
      *(_QWORD *)(v6 + 24) = sub_C7D670(v54, 4);
      *(_DWORD *)(v6 + 32) = v66;
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(v6 + 24), 8 * v10, 4);
      *(_BYTE *)(v6 + 16) |= 1u;
    }
LABEL_106:
    sub_37BEA10(v6 + 8);
    goto LABEL_10;
  }
  v11 = *(_DWORD **)(v6 + 24);
  v12 = 2 * v10;
LABEL_7:
  for ( i = &v11[v12]; i != v11; v11 += 2 )
    *v11 = -1;
  *(_QWORD *)(v6 + 16) &= 1uLL;
LABEL_10:
  v14 = *(_DWORD *)(v6 + 688);
  ++*(_QWORD *)(v6 + 680);
  *(_DWORD *)(v6 + 96) = 0;
  v15 = v14 >> 1;
  if ( v15 )
  {
    if ( (*(_BYTE *)(v6 + 688) & 1) == 0 )
    {
      v16 = 4 * v15;
      goto LABEL_13;
    }
LABEL_66:
    v18 = (_DWORD *)(v6 + 696);
    v19 = 32;
    goto LABEL_16;
  }
  if ( !*(_DWORD *)(v6 + 692) )
    goto LABEL_19;
  v16 = 0;
  if ( (*(_BYTE *)(v6 + 688) & 1) != 0 )
    goto LABEL_66;
LABEL_13:
  v17 = *(unsigned int *)(v6 + 704);
  if ( (unsigned int)v17 > v16 && (unsigned int)v17 > 0x40 )
  {
    if ( v15 && (v56 = v15 - 1) != 0 )
    {
      _BitScanReverse(&v56, v56);
      v57 = 1 << (33 - (v56 ^ 0x1F));
      if ( (unsigned int)(v57 - 9) > 0x36 )
      {
        if ( (_DWORD)v17 == v57 )
          goto LABEL_98;
        v68 = 1 << (33 - (v56 ^ 0x1F));
        sub_C7D6A0(*(_QWORD *)(v6 + 696), 16 * v17, 8);
        v59 = v68;
        v60 = *(_BYTE *)(v6 + 688) | 1;
        *(_BYTE *)(v6 + 688) = v60;
        if ( v68 <= 8 )
          goto LABEL_98;
        v58 = 16LL * v68;
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(v6 + 696), 16 * v17, 8);
        v58 = 1024;
        v59 = 64;
        v60 = *(_BYTE *)(v6 + 688);
      }
      v67 = v59;
      *(_BYTE *)(v6 + 688) = v60 & 0xFE;
      *(_QWORD *)(v6 + 696) = sub_C7D670(v58, 8);
      *(_DWORD *)(v6 + 704) = v67;
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(v6 + 696), 16 * v17, 8);
      *(_BYTE *)(v6 + 688) |= 1u;
    }
LABEL_98:
    sub_37BEA60(v6 + 680);
    goto LABEL_19;
  }
  v18 = *(_DWORD **)(v6 + 696);
  v19 = 4 * v17;
LABEL_16:
  for ( j = &v18[v19]; j != v18; v18 += 4 )
    *v18 = -1;
  *(_QWORD *)(v6 + 688) &= 1uLL;
LABEL_19:
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 408LL) + 304LL) = 0;
  v21 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 408LL);
  v22 = *(_QWORD **)(**(_QWORD **)(a1 + 16) + 8LL * *(int *)(a2 + 24));
  v23 = *(unsigned int *)(v21 + 40);
  *(_DWORD *)(v21 + 280) = v5;
  v24 = 0;
  if ( (_DWORD)v23 )
  {
    do
    {
      *(_QWORD *)(*(_QWORD *)(v21 + 32) + v24) = *(_QWORD *)(*v22 + v24);
      v24 += 8;
    }
    while ( v24 != 8 * v23 );
  }
  v25 = a2 + 48;
  v26 = 592 * v4;
  sub_37E1AE0(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 432LL),
    a2,
    *(_QWORD **)(**(_QWORD **)(a1 + 16) + 8LL * *(int *)(a2 + 24)),
    (_QWORD *)(*(_QWORD *)(a1 + 8) + 2168LL),
    v26 + **(_QWORD **)(a1 + 24),
    (_QWORD *)**(unsigned int **)(a1 + 32),
    a3);
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 416LL) = v5;
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 420LL) = 1;
  v27 = *(_QWORD *)(a2 + 56);
  if ( v27 != a2 + 48 )
  {
    do
    {
      while ( 1 )
      {
        sub_37E06C0(*(_QWORD *)(a1 + 8), v27, *(_QWORD **)(a1 + 40), *(_QWORD **)(a1 + 16));
        sub_37D3140(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 432LL), *(_DWORD *)(*(_QWORD *)(a1 + 8) + 420LL), v27);
        ++*(_DWORD *)(*(_QWORD *)(a1 + 8) + 420LL);
        if ( !v27 )
          BUG();
        if ( (*(_BYTE *)v27 & 4) == 0 )
          break;
        v27 = *(_QWORD *)(v27 + 8);
        if ( v27 == v25 )
          goto LABEL_27;
      }
      while ( (*(_BYTE *)(v27 + 44) & 8) != 0 )
        v27 = *(_QWORD *)(v27 + 8);
      v27 = *(_QWORD *)(v27 + 8);
    }
    while ( v27 != v25 );
  }
LABEL_27:
  v28 = (unsigned __int64 **)(**(_QWORD **)(a1 + 16) + 8LL * *(int *)(a2 + 24));
  v29 = *v28;
  *v28 = 0;
  if ( v29 )
  {
    if ( (unsigned __int64 *)*v29 != v29 + 2 )
      _libc_free(*v29);
    j_j___libc_free_0((unsigned __int64)v29);
  }
  v30 = (unsigned __int64 **)(**(_QWORD **)(a1 + 40) + 8LL * *(int *)(a2 + 24));
  v31 = *v30;
  *v30 = 0;
  if ( v31 )
  {
    if ( (unsigned __int64 *)*v31 != v31 + 2 )
      _libc_free(*v31);
    j_j___libc_free_0((unsigned __int64)v31);
  }
  *(_DWORD *)(**(_QWORD **)(a1 + 24) + v26 + 8) = 0;
  v32 = **(_QWORD **)a1 + v70;
  v33 = *(_DWORD *)(v32 + 16);
  ++*(_QWORD *)(v32 + 8);
  v34 = v33 >> 1;
  if ( v34 )
  {
    if ( (*(_BYTE *)(v32 + 16) & 1) == 0 )
    {
      v35 = 4 * v34;
      goto LABEL_38;
    }
LABEL_67:
    v37 = (_DWORD *)(v32 + 24);
    v38 = 16;
    goto LABEL_41;
  }
  if ( !*(_DWORD *)(v32 + 20) )
    goto LABEL_44;
  v35 = 0;
  if ( (*(_BYTE *)(v32 + 16) & 1) != 0 )
    goto LABEL_67;
LABEL_38:
  v36 = *(unsigned int *)(v32 + 32);
  if ( (unsigned int)v36 > v35 && (unsigned int)v36 > 0x40 )
  {
    if ( v34 && (v61 = v34 - 1) != 0 )
    {
      _BitScanReverse(&v61, v61);
      v62 = 1 << (33 - (v61 ^ 0x1F));
      if ( v62 - 9 > 0x36 )
      {
        if ( (_DWORD)v36 == v62 )
          goto LABEL_91;
        sub_C7D6A0(*(_QWORD *)(v32 + 24), 8 * v36, 4);
        v63 = *(_BYTE *)(v32 + 16) | 1;
        *(_BYTE *)(v32 + 16) = v63;
        if ( v62 <= 8 )
          goto LABEL_91;
        v64 = 8LL * v62;
      }
      else
      {
        v62 = 64;
        sub_C7D6A0(*(_QWORD *)(v32 + 24), 8 * v36, 4);
        v63 = *(_BYTE *)(v32 + 16);
        v64 = 512;
      }
      *(_BYTE *)(v32 + 16) = v63 & 0xFE;
      v65 = sub_C7D670(v64, 4);
      *(_DWORD *)(v32 + 32) = v62;
      *(_QWORD *)(v32 + 24) = v65;
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(v32 + 24), 8 * v36, 4);
      *(_BYTE *)(v32 + 16) |= 1u;
    }
LABEL_91:
    sub_37BEA10(v32 + 8);
    goto LABEL_44;
  }
  v37 = *(_DWORD **)(v32 + 24);
  v38 = 2 * v36;
LABEL_41:
  for ( k = &v37[v38]; k != v37; v37 += 2 )
    *v37 = -1;
  *(_QWORD *)(v32 + 16) &= 1uLL;
LABEL_44:
  v40 = *(_DWORD *)(v32 + 688);
  ++*(_QWORD *)(v32 + 680);
  *(_DWORD *)(v32 + 96) = 0;
  result = (_DWORD *)(v40 >> 1);
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(v32 + 688) & 1) == 0 )
    {
      v42 = 4 * (_DWORD)result;
      goto LABEL_47;
    }
LABEL_68:
    result = (_DWORD *)(v32 + 696);
    v44 = 32;
    goto LABEL_50;
  }
  if ( !*(_DWORD *)(v32 + 692) )
    return result;
  v42 = 0;
  if ( (*(_BYTE *)(v32 + 688) & 1) != 0 )
    goto LABEL_68;
LABEL_47:
  v43 = *(unsigned int *)(v32 + 704);
  if ( v42 < (unsigned int)v43 && (unsigned int)v43 > 0x40 )
  {
    if ( (_DWORD)result && (v46 = (_DWORD)result - 1) != 0 )
    {
      _BitScanReverse(&v46, v46);
      v47 = 1 << (33 - (v46 ^ 0x1F));
      if ( v47 - 9 > 0x36 )
      {
        if ( (_DWORD)v43 == v47 )
          return sub_37BEA60(v32 + 680);
        sub_C7D6A0(*(_QWORD *)(v32 + 696), 16LL * *(unsigned int *)(v32 + 704), 8);
        v49 = *(_BYTE *)(v32 + 688) | 1;
        *(_BYTE *)(v32 + 688) = v49;
        if ( v47 <= 8 )
          return sub_37BEA60(v32 + 680);
        v48 = 16LL * v47;
      }
      else
      {
        v47 = 64;
        sub_C7D6A0(*(_QWORD *)(v32 + 696), 16LL * *(unsigned int *)(v32 + 704), 8);
        v48 = 1024;
        v49 = *(_BYTE *)(v32 + 688);
      }
      *(_BYTE *)(v32 + 688) = v49 & 0xFE;
      v50 = sub_C7D670(v48, 8);
      *(_DWORD *)(v32 + 704) = v47;
      *(_QWORD *)(v32 + 696) = v50;
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(v32 + 696), 16LL * *(unsigned int *)(v32 + 704), 8);
      *(_BYTE *)(v32 + 688) |= 1u;
    }
    return sub_37BEA60(v32 + 680);
  }
  result = *(_DWORD **)(v32 + 696);
  v44 = 4 * v43;
LABEL_50:
  for ( m = &result[v44]; m != result; result += 4 )
    *result = -1;
  *(_QWORD *)(v32 + 688) &= 1uLL;
  return result;
}
