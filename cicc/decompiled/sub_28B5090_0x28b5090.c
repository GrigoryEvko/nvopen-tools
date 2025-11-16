// Function: sub_28B5090
// Address: 0x28b5090
//
unsigned __int64 *__fastcall sub_28B5090(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // r8
  __int64 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 *v19; // r9
  __int64 v20; // rax
  int v21; // r9d
  int v22; // r9d
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // r10
  __int64 v25; // r15
  int v26; // r14d
  __int64 i; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // rcx
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rdi
  __int64 *v34; // r9
  __int64 *v35; // r11
  __int64 *v36; // r8
  __int64 *v37; // rdi
  unsigned int v38; // r14d
  int v39; // edi
  int v40; // r10d
  __int64 v41; // r10
  __int64 v42; // r9
  __int64 v43; // r15
  __int64 v44; // rdi
  int v45; // r14d
  char v46; // di
  int v47; // edi
  int v48; // edi
  unsigned __int64 j; // r13
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  __int64 *v53; // r8
  __int64 *v54; // rdi
  __int64 v55; // r10
  __int64 *v56; // r9
  __int64 *v57; // rax
  __int64 v58; // r11
  int v59; // edi
  __int64 v60; // r9
  int v61; // r8d
  __int64 v62; // rax
  int v63; // r9d
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // [rsp+10h] [rbp-50h]
  __int64 v67; // [rsp+20h] [rbp-40h]
  __int64 v68; // [rsp+28h] [rbp-38h]
  unsigned __int64 v69; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 6);
  if ( v6 == 0xAAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - v5) >> 6);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * ((__int64)(v4 - v5) >> 6);
  v11 = a2 - v5;
  if ( v9 )
  {
    v64 = 0x7FFFFFFFFFFFFF80LL;
LABEL_63:
    v69 = v64;
    v67 = sub_22077B0(v64);
    v65 = v67 + v69;
    v68 = v67 + 192;
    goto LABEL_7;
  }
  if ( v10 )
  {
    if ( v10 > 0xAAAAAAAAAAAAAALL )
      v10 = 0xAAAAAAAAAAAAAALL;
    v64 = 192 * v10;
    goto LABEL_63;
  }
  v68 = 192;
  v65 = 0;
  v67 = 0;
LABEL_7:
  if ( v67 + v11 )
    sub_28B4EA0(v67 + v11, a3);
  if ( a2 != v5 )
  {
    v12 = v67;
    v13 = v5 + 24;
    v14 = v5;
    v15 = (__int64 *)(v67 + 88);
    v16 = v67 + 24;
    while ( !v12 )
    {
LABEL_23:
      v14 += 192LL;
      v12 += 192;
      v15 += 24;
      v16 += 192;
      v13 += 192LL;
      if ( a2 == v14 )
      {
        v68 = v67 + ((3 * ((0x2AAAAAAAAAAAAABLL * ((a2 - 192 - v5) >> 6)) & 0x3FFFFFFFFFFFFFFLL) + 6) << 6);
        goto LABEL_25;
      }
    }
    v17 = *(_QWORD *)(v13 - 24);
    *(_QWORD *)(v16 - 16) = 0;
    v18 = v12 + 8;
    v19 = (__int64 *)v16;
    *(_QWORD *)(v16 - 24) = v17;
    v20 = v14 + 8;
    *(_DWORD *)(v12 + 16) = 1;
    *(_DWORD *)(v16 - 4) = 0;
    do
    {
      if ( v19 )
        *v19 = -4096;
      ++v19;
    }
    while ( v19 != v15 );
    v21 = *(_DWORD *)(v14 + 16);
    *(_DWORD *)(v14 + 16) = *(_DWORD *)(v12 + 16) & 0xFFFFFFFE | v21 & 1;
    *(_DWORD *)(v12 + 16) = v21 & 0xFFFFFFFE | *(_DWORD *)(v12 + 16) & 1;
    v22 = *(_DWORD *)(v16 - 4);
    *(_DWORD *)(v16 - 4) = *(_DWORD *)(v13 - 4);
    *(_DWORD *)(v13 - 4) = v22;
    if ( (*(_BYTE *)(v12 + 16) & 1) != 0 )
    {
      v23 = v16;
      v24 = v13;
      if ( (*(_BYTE *)(v14 + 16) & 1) != 0 )
      {
        v56 = (__int64 *)v13;
        v57 = (__int64 *)v16;
        do
        {
          v58 = *v57;
          *v57++ = *v56;
          *v56++ = v58;
        }
        while ( v57 != v15 );
        goto LABEL_22;
      }
    }
    else
    {
      if ( (*(_BYTE *)(v14 + 16) & 1) == 0 )
      {
        v62 = *(_QWORD *)v16;
        *(_QWORD *)v16 = *(_QWORD *)v13;
        v63 = *(_DWORD *)(v13 + 8);
        *(_QWORD *)v13 = v62;
        LODWORD(v62) = *(_DWORD *)(v16 + 8);
        *(_DWORD *)(v16 + 8) = v63;
        *(_DWORD *)(v13 + 8) = v62;
LABEL_22:
        *(_BYTE *)(v16 + 64) = *(_BYTE *)(v13 + 64);
        *(_DWORD *)(v16 + 68) = *(_DWORD *)(v13 + 68);
        *(_QWORD *)(v16 + 72) = *(_QWORD *)(v13 + 72);
        *(_QWORD *)(v16 + 80) = *(_QWORD *)(v13 + 80);
        *(_DWORD *)(v16 + 88) = *(_DWORD *)(v13 + 88);
        *(_DWORD *)(v16 + 104) = *(_DWORD *)(v13 + 104);
        *(_QWORD *)(v16 + 96) = *(_QWORD *)(v13 + 96);
        v28 = *(_QWORD *)(v13 + 112);
        *(_DWORD *)(v13 + 104) = 0;
        *(_QWORD *)(v16 + 112) = v28;
        *(_QWORD *)(v16 + 120) = *(_QWORD *)(v13 + 120);
        *(_DWORD *)(v16 + 128) = *(_DWORD *)(v13 + 128);
        *(_DWORD *)(v16 + 144) = *(_DWORD *)(v13 + 144);
        *(_QWORD *)(v16 + 136) = *(_QWORD *)(v13 + 136);
        LODWORD(v28) = *(_DWORD *)(v13 + 152);
        *(_DWORD *)(v13 + 144) = 0;
        *(_DWORD *)(v16 + 152) = v28;
        *(_QWORD *)(v16 + 160) = *(_QWORD *)(v13 + 160);
        goto LABEL_23;
      }
      v23 = v13;
      v20 = v12 + 8;
      v24 = v16;
      v18 = v14 + 8;
    }
    v25 = *(_QWORD *)(v20 + 16);
    v26 = *(_DWORD *)(v20 + 24);
    *(_BYTE *)(v20 + 8) |= 1u;
    for ( i = 0; i != 64; i += 8 )
      *(_QWORD *)(v24 + i) = *(_QWORD *)(v23 + i);
    *(_BYTE *)(v18 + 8) &= ~1u;
    *(_QWORD *)(v18 + 16) = v25;
    *(_DWORD *)(v18 + 24) = v26;
    goto LABEL_22;
  }
LABEL_25:
  if ( a2 == v4 )
    goto LABEL_39;
  v29 = a2;
  v30 = (__int64 *)(v68 + 88);
  v31 = v68 + 24;
  v32 = (__int64 *)(a2 + 24);
  do
  {
    v33 = *(v32 - 3);
    v34 = v30 - 11;
    *(_QWORD *)(v31 - 16) = 0;
    v35 = v30 - 10;
    v36 = (__int64 *)(v29 + 8);
    *(_QWORD *)(v31 - 24) = v33;
    v37 = (__int64 *)v31;
    *((_DWORD *)v30 - 18) = 1;
    *(_DWORD *)(v31 - 4) = 0;
    do
    {
      if ( v37 )
        *v37 = -4096;
      ++v37;
    }
    while ( v37 != v30 );
    v38 = v34[2] & 0xFFFFFFFE | *(_DWORD *)(v29 + 16) & 1;
    *((_DWORD *)v34 + 4) = *(_DWORD *)(v29 + 16) & 0xFFFFFFFE | v34[2] & 1;
    v39 = *(_DWORD *)(v31 - 4);
    *(_DWORD *)(v29 + 16) = v38;
    v40 = *((_DWORD *)v32 - 1);
    *((_DWORD *)v32 - 1) = v39;
    *(_DWORD *)(v31 - 4) = v40;
    if ( (v34[2] & 1) == 0 )
    {
      if ( (*(_BYTE *)(v29 + 16) & 1) == 0 )
      {
        v59 = *(_DWORD *)(v31 + 8);
        v60 = *v32;
        *v32 = *(_QWORD *)v31;
        v61 = *((_DWORD *)v32 + 2);
        *(_QWORD *)v31 = v60;
        *(_DWORD *)(v31 + 8) = v61;
        *((_DWORD *)v32 + 2) = v59;
        goto LABEL_37;
      }
      v41 = (__int64)v32;
      v36 = v30 - 10;
      v42 = v31;
      v35 = (__int64 *)(v29 + 8);
LABEL_34:
      *((_BYTE *)v36 + 8) |= 1u;
      v43 = v36[2];
      v44 = 0;
      v45 = *((_DWORD *)v36 + 6);
      do
      {
        *(_QWORD *)(v42 + v44) = *(_QWORD *)(v41 + v44);
        v44 += 8;
      }
      while ( v44 != 64 );
      *((_BYTE *)v35 + 8) &= ~1u;
      v35[2] = v43;
      *((_DWORD *)v35 + 6) = v45;
      goto LABEL_37;
    }
    v41 = v31;
    v42 = (__int64)v32;
    if ( (*(_BYTE *)(v29 + 16) & 1) == 0 )
      goto LABEL_34;
    v53 = v32;
    v54 = (__int64 *)v31;
    do
    {
      v55 = *v54;
      *v54++ = *v53;
      *v53++ = v55;
    }
    while ( v30 != v54 );
LABEL_37:
    v46 = *((_BYTE *)v32 + 64);
    v29 += 192;
    v30 += 24;
    v31 += 192;
    v32 += 24;
    *(_BYTE *)(v31 - 128) = v46;
    *(_DWORD *)(v31 - 124) = *((_DWORD *)v32 - 31);
    *(_QWORD *)(v31 - 120) = *(v32 - 15);
    *(_QWORD *)(v31 - 112) = *(v32 - 14);
    *(_DWORD *)(v31 - 104) = *((_DWORD *)v32 - 26);
    v47 = *((_DWORD *)v32 - 22);
    *((_DWORD *)v32 - 22) = 0;
    *(_DWORD *)(v31 - 88) = v47;
    *(_QWORD *)(v31 - 96) = *(v32 - 12);
    *(_QWORD *)(v31 - 80) = *(v32 - 10);
    *(_QWORD *)(v31 - 72) = *(v32 - 9);
    *(_DWORD *)(v31 - 64) = *((_DWORD *)v32 - 16);
    v48 = *((_DWORD *)v32 - 12);
    *((_DWORD *)v32 - 12) = 0;
    *(_DWORD *)(v31 - 48) = v48;
    *(_QWORD *)(v31 - 56) = *(v32 - 7);
    *(_DWORD *)(v31 - 40) = *((_DWORD *)v32 - 10);
    *(_QWORD *)(v31 - 32) = *(v32 - 4);
  }
  while ( v4 != v29 );
  v68 += (3 * ((0x2AAAAAAAAAAAAABLL * ((v4 - a2 - 192) >> 6)) & 0x3FFFFFFFFFFFFFFLL) + 3) << 6;
LABEL_39:
  for ( j = v5; j != v4; j += 192LL )
  {
    if ( *(_DWORD *)(j + 168) > 0x40u )
    {
      v50 = *(_QWORD *)(j + 160);
      if ( v50 )
        j_j___libc_free_0_0(v50);
    }
    if ( *(_DWORD *)(j + 128) > 0x40u )
    {
      v51 = *(_QWORD *)(j + 120);
      if ( v51 )
        j_j___libc_free_0_0(v51);
    }
    if ( (*(_BYTE *)(j + 16) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(j + 24), 8LL * *(unsigned int *)(j + 32), 8);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  *a1 = v67;
  a1[1] = v68;
  a1[2] = v65;
  return a1;
}
