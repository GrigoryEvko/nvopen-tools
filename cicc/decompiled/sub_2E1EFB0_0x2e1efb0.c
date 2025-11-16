// Function: sub_2E1EFB0
// Address: 0x2e1efb0
//
__int64 __fastcall sub_2E1EFB0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6, _QWORD *a7)
{
  unsigned int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 *v14; // r13
  __int64 v15; // rdx
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // r13
  unsigned __int64 v26; // rdx
  signed __int64 v27; // rcx
  __int64 v28; // rdx
  _QWORD *v29; // r9
  unsigned int v30; // ecx
  __int64 v31; // rsi
  __int64 *v32; // rdi
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 *v37; // r13
  __int64 *v38; // rbx
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 *j; // rdi
  __int64 v42; // rax
  __int64 *v43; // rdx
  __int64 *i; // rdi
  __int64 v45; // rax
  __int64 *v46; // rdx
  __int64 *v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // [rsp+8h] [rbp-B8h]
  char v50; // [rsp+14h] [rbp-ACh]
  __int64 v54; // [rsp+30h] [rbp-90h]
  __int64 v56; // [rsp+40h] [rbp-80h]
  unsigned int v57; // [rsp+48h] [rbp-78h]
  int v58; // [rsp+4Ch] [rbp-74h]
  int v59; // [rsp+5Ch] [rbp-64h] BYREF
  __int64 v60; // [rsp+60h] [rbp-60h] BYREF
  __int64 v61; // [rsp+68h] [rbp-58h]
  __int64 v62; // [rsp+70h] [rbp-50h]
  __int64 v63; // [rsp+78h] [rbp-48h]
  _BYTE *v64; // [rsp+80h] [rbp-40h]
  __int64 v65; // [rsp+88h] [rbp-38h]
  _BYTE v66[48]; // [rsp+90h] [rbp-30h] BYREF

  v8 = *(_DWORD *)(a5 + 24);
  v9 = 8LL * (v8 >> 6);
  v50 = v8 & 0x3F;
  v49 = v9;
  v10 = (*(_QWORD *)(*a6 + v9) >> v8) & 1LL;
  if ( ((*(_QWORD *)(*a6 + v9) >> v8) & 1) != 0 )
    return (unsigned int)v10;
  v11 = (__int64 *)(v9 + *a7);
  v12 = *v11;
  if ( _bittest64(&v12, v8 & 0x3F) )
    return (unsigned int)v10;
  v14 = *(__int64 **)(a5 + 64);
  v15 = *(unsigned int *)(a5 + 72);
  v64 = v66;
  v16 = &v14[v15];
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0;
  if ( v14 == v16 )
  {
LABEL_26:
    LODWORD(v10) = 0;
    *v11 |= 1LL << v50;
    goto LABEL_37;
  }
  do
  {
    v17 = *v14++;
    v59 = *(_DWORD *)(v17 + 24);
    sub_2E1EA30((__int64)&v60, &v59);
  }
  while ( v16 != v14 );
  if ( !(_DWORD)v65 )
  {
LABEL_25:
    v11 = (__int64 *)(*a7 + v49);
    goto LABEL_26;
  }
  v58 = 0;
  v18 = 0;
  while ( 1 )
  {
    v19 = *(unsigned int *)&v64[4 * v18];
    v20 = (unsigned int)v19 >> 6;
    v56 = 8 * v20;
    v54 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8 * v19);
    v21 = *(_QWORD *)(a1[5] + 8 * v20);
    v57 = *(_DWORD *)&v64[4 * v18] & 0x3F;
    if ( _bittest64(&v21, v19) )
    {
      v22 = *(__int64 **)(a1[18] + 16LL * *(unsigned int *)(v54 + 24));
      if ( v22 )
      {
        if ( v22 != &qword_501EAE0 )
        {
          v43 = *(__int64 **)(v54 + 112);
          for ( i = &v43[*(unsigned int *)(v54 + 120)];
                i != v43;
                *(_QWORD *)(*a6 + 8LL * (*(_DWORD *)(v45 + 24) >> 6)) |= 1LL << *(_DWORD *)(v45 + 24) )
          {
            v45 = *v43++;
          }
          goto LABEL_35;
        }
      }
    }
    v23 = (__int64 *)(*(_QWORD *)(a1[2] + 152LL) + 16LL * *(unsigned int *)(v54 + 24));
    v24 = v23[1];
    v25 = *v23;
    v26 = v24 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = ((v24 >> 1) & 3) != 0 ? v26 | (2LL * (int)(((v24 >> 1) & 3) - 1)) : *(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v28 = *(unsigned int *)(a2 + 8);
    if ( !(3 * v28) )
      break;
    v29 = *(_QWORD **)a2;
    v30 = *(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v27 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v31 = v28 >> 1;
        v32 = &v29[(v28 >> 1) + (v28 & 0xFFFFFFFFFFFFFFFELL)];
        if ( v30 < (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) )
          break;
        v29 = v32 + 3;
        v28 = v28 - v31 - 1;
        if ( v28 <= 0 )
          goto LABEL_18;
      }
      v28 >>= 1;
    }
    while ( v31 > 0 );
LABEL_18:
    if ( *(_QWORD **)a2 == v29
      || (*(_DWORD *)((*(v29 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v29 - 2) >> 1) & 3) <= (*(_DWORD *)((v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v25 >> 1) & 3) )
    {
      break;
    }
    if ( !sub_2E1D650(a3, a4, *(v29 - 2), v24) )
    {
      v40 = *(__int64 **)(v54 + 112);
      for ( j = &v40[*(unsigned int *)(v54 + 120)];
            j != v40;
            *(_QWORD *)(*a6 + 8LL * (*(_DWORD *)(v42 + 24) >> 6)) |= 1LL << *(_DWORD *)(v42 + 24) )
      {
        v42 = *v40++;
      }
LABEL_35:
      *(_QWORD *)(*a6 + v49) |= 1LL << v50;
      goto LABEL_36;
    }
LABEL_24:
    v18 = (unsigned int)++v58;
    if ( v58 == (_DWORD)v65 )
      goto LABEL_25;
  }
  v33 = (__int64 *)(*a7 + v56);
  v34 = *v33;
  if ( _bittest64(&v34, v57) )
  {
LABEL_23:
    *v33 = v34 | (1LL << v57);
    goto LABEL_24;
  }
  if ( sub_2E1D650(a3, a4, v25, v24) )
  {
    v33 = (__int64 *)(*a7 + v56);
    v34 = *v33;
    goto LABEL_23;
  }
  v35 = *a6;
  v36 = *(_QWORD *)(*a6 + 8 * v20);
  if ( !_bittest64(&v36, v57) )
  {
    v37 = *(__int64 **)(v54 + 64);
    v38 = &v37[*(unsigned int *)(v54 + 72)];
    while ( v38 != v37 )
    {
      v39 = *v37++;
      v59 = *(_DWORD *)(v39 + 24);
      sub_2E1EA30((__int64)&v60, &v59);
    }
    goto LABEL_24;
  }
  v46 = *(__int64 **)(v54 + 112);
  v47 = &v46[*(unsigned int *)(v54 + 120)];
  if ( v46 != v47 )
  {
    while ( 1 )
    {
      v48 = *v46++;
      *(_QWORD *)(v35 + 8LL * (*(_DWORD *)(v48 + 24) >> 6)) |= 1LL << *(_DWORD *)(v48 + 24);
      if ( v47 == v46 )
        break;
      v35 = *a6;
    }
    v35 = *a6;
  }
  *(_QWORD *)(v35 + v49) |= 1LL << v50;
LABEL_36:
  LODWORD(v10) = 1;
LABEL_37:
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  sub_C7D6A0(v61, 4LL * (unsigned int)v63, 4);
  return (unsigned int)v10;
}
