// Function: sub_1DC4D00
// Address: 0x1dc4d00
//
__int64 __fastcall sub_1DC4D00(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6, _QWORD *a7)
{
  unsigned int v9; // ecx
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v15; // r13
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rdx
  _QWORD *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // r12d
  unsigned __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 *v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // r11
  unsigned __int64 v30; // rdi
  signed __int64 v31; // rdi
  _QWORD *v32; // r9
  __int64 v33; // rdx
  unsigned int v34; // edi
  __int64 v35; // rcx
  __int64 *v36; // rsi
  __int64 *v37; // r8
  __int64 v38; // r9
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // r12
  __int64 *v43; // r13
  __int64 v44; // rax
  __int64 *v45; // rdi
  __int64 *j; // rdx
  __int64 v47; // rax
  __int64 *v48; // rdi
  __int64 *i; // rdx
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 *v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // [rsp+8h] [rbp-D8h]
  char v55; // [rsp+14h] [rbp-CCh]
  __int64 *v58; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+38h] [rbp-A8h]
  __int64 *v61; // [rsp+38h] [rbp-A8h]
  __int64 v62; // [rsp+40h] [rbp-A0h]
  unsigned int v63; // [rsp+48h] [rbp-98h]
  __int64 v64; // [rsp+50h] [rbp-90h]
  _QWORD *v65; // [rsp+58h] [rbp-88h]
  int v66; // [rsp+58h] [rbp-88h]
  unsigned int v67; // [rsp+6Ch] [rbp-74h] BYREF
  __int64 v68; // [rsp+70h] [rbp-70h] BYREF
  __int64 v69; // [rsp+78h] [rbp-68h]
  __int64 v70; // [rsp+80h] [rbp-60h]
  __int64 v71; // [rsp+88h] [rbp-58h]
  __int64 v72; // [rsp+90h] [rbp-50h]
  __int64 v73; // [rsp+98h] [rbp-48h]
  __int64 v74; // [rsp+A0h] [rbp-40h]

  v9 = *(_DWORD *)(a5 + 48);
  v10 = 8LL * (v9 >> 6);
  v55 = v9 & 0x3F;
  v54 = v10;
  v11 = (*(_QWORD *)(*a6 + v10) >> v9) & 1LL;
  if ( ((*(_QWORD *)(*a6 + v10) >> v9) & 1) != 0 )
    return (unsigned int)v11;
  v12 = (__int64 *)(v10 + *a7);
  v13 = *v12;
  if ( _bittest64(&v13, v9 & 0x3F) )
    return (unsigned int)v11;
  v15 = *(__int64 **)(a5 + 72);
  v16 = *(__int64 **)(a5 + 64);
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  if ( v15 == v16 )
  {
LABEL_26:
    LODWORD(v11) = 0;
    *v12 |= 1LL << v55;
    goto LABEL_37;
  }
  v65 = a1;
  v17 = v16;
  do
  {
    v18 = *v17++;
    v67 = *(_DWORD *)(v18 + 48);
    sub_1DC4AC0((__int64)&v68, &v67);
  }
  while ( v15 != v17 );
  v19 = v65;
  v64 = v72;
  if ( v72 == v73 )
  {
LABEL_25:
    v12 = (__int64 *)(*a7 + v54);
    goto LABEL_26;
  }
  v20 = 0;
  v66 = 0;
  v58 = &a3[a4];
  while ( 1 )
  {
    v21 = *(_DWORD *)(v64 + 4 * v20);
    v22 = v21;
    v23 = v21 >> 6;
    v24 = *(_QWORD *)(*(_QWORD *)(*v19 + 96LL) + 8 * v22);
    v60 = 8 * v23;
    v63 = v22 & 0x3F;
    v25 = *(_QWORD *)(v19[5] + 8 * v23);
    if ( _bittest64(&v25, v22) )
    {
      v26 = *(__int64 **)(v19[12] + 16LL * *(unsigned int *)(v24 + 48));
      if ( v26 )
      {
        if ( v26 != &qword_4FC4510 )
        {
          v48 = *(__int64 **)(v24 + 96);
          for ( i = *(__int64 **)(v24 + 88);
                i != v48;
                *(_QWORD *)(*a6 + 8LL * (*(_DWORD *)(v50 + 48) >> 6)) |= 1LL << *(_DWORD *)(v50 + 48) )
          {
            v50 = *i++;
          }
          goto LABEL_35;
        }
      }
    }
    v27 = (__int64 *)(*(_QWORD *)(v19[2] + 392LL) + 16LL * *(unsigned int *)(v24 + 48));
    v28 = v27[1];
    v29 = *v27;
    v30 = v28 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = ((v28 >> 1) & 3) != 0 ? (2LL * (int)(((v28 >> 1) & 3) - 1)) | v30 : *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v32 = *(_QWORD **)a2;
    v33 = *(unsigned int *)(a2 + 8);
    if ( !(3 * v33) )
      break;
    v34 = *(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v31 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v35 = v33 >> 1;
        v36 = &v32[(v33 >> 1) + (v33 & 0xFFFFFFFFFFFFFFFELL)];
        if ( v34 < (*(_DWORD *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v36 >> 1) & 3) )
          break;
        v32 = v36 + 3;
        v33 = v33 - v35 - 1;
        if ( v33 <= 0 )
          goto LABEL_19;
      }
      v33 >>= 1;
    }
    while ( v35 > 0 );
LABEL_19:
    if ( *(_QWORD **)a2 == v32
      || (*(_DWORD *)((*(v32 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v32 - 2) >> 1) & 3) <= (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v29 >> 1) & 3) )
    {
      break;
    }
    if ( v58 == sub_1DC3450(a3, (__int64)v58, *(v32 - 2), v28) )
    {
      v45 = *(__int64 **)(v24 + 96);
      for ( j = *(__int64 **)(v24 + 88);
            v45 != j;
            *(_QWORD *)(*a6 + 8LL * (*(_DWORD *)(v47 + 48) >> 6)) |= 1LL << *(_DWORD *)(v47 + 48) )
      {
        v47 = *j++;
      }
LABEL_35:
      *(_QWORD *)(*a6 + v54) |= 1LL << v55;
      goto LABEL_36;
    }
LABEL_24:
    v20 = (unsigned int)++v66;
    if ( v66 == (v73 - v64) >> 2 )
      goto LABEL_25;
  }
  v37 = (__int64 *)(*a7 + v60);
  v38 = *v37;
  if ( _bittest64(&v38, v63)
    || (v61 = (__int64 *)(*a7 + v60),
        v62 = *v37,
        v39 = sub_1DC3450(a3, (__int64)v58, v29, v28),
        v38 = v62,
        v37 = v61,
        v58 != v39) )
  {
    *v37 = v38 | (1LL << v63);
    v64 = v72;
    goto LABEL_24;
  }
  v40 = *a6;
  v41 = *(_QWORD *)(*a6 + 8 * v23);
  if ( !_bittest64(&v41, v63) )
  {
    v42 = *(__int64 **)(v24 + 72);
    v43 = *(__int64 **)(v24 + 64);
    if ( v42 != v43 )
    {
      do
      {
        v44 = *v43++;
        v67 = *(_DWORD *)(v44 + 48);
        sub_1DC4AC0((__int64)&v68, &v67);
      }
      while ( v42 != v43 );
      v64 = v72;
    }
    goto LABEL_24;
  }
  v51 = *(__int64 **)(v24 + 96);
  v52 = *(__int64 **)(v24 + 88);
  if ( v51 != v52 )
  {
    while ( 1 )
    {
      v53 = *v52++;
      *(_QWORD *)(v40 + 8LL * (*(_DWORD *)(v53 + 48) >> 6)) |= 1LL << *(_DWORD *)(v53 + 48);
      if ( v51 == v52 )
        break;
      v40 = *a6;
    }
    v40 = *a6;
  }
  *(_QWORD *)(v40 + v54) |= 1LL << v55;
LABEL_36:
  LODWORD(v11) = 1;
LABEL_37:
  if ( v72 )
    j_j___libc_free_0(v72, v74 - v72);
  j___libc_free_0(v69);
  return (unsigned int)v11;
}
