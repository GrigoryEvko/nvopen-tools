// Function: sub_14813B0
// Address: 0x14813b0
//
__int64 __fastcall sub_14813B0(_QWORD *a1, __int64 **a2, __m128i a3, __m128i a4)
{
  __int64 *v6; // r15
  __int64 v7; // r14
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rbx
  const void *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  __int64 v20; // rdx
  unsigned int v21; // ebx
  unsigned int v22; // esi
  unsigned int v23; // edx
  __int64 v24; // rbx
  unsigned int v25; // r11d
  __int64 v26; // rax
  unsigned __int64 v27; // r9
  __int64 v28; // rbx
  __int64 *v29; // rdi
  __int64 v30; // r14
  unsigned __int16 v31; // dx
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  const void *v35; // r9
  signed __int64 v36; // r14
  __int64 v37; // r15
  unsigned int v38; // r15d
  unsigned int v39; // ebx
  __int64 v40; // rcx
  const void *v41; // rsi
  __int64 v42; // r14
  const void *v43; // rax
  size_t v44; // r14
  void *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r15
  unsigned __int64 v49; // r15
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 *v56; // rsi
  __int64 *v57; // r15
  __int64 v58; // r14
  __int64 *v59; // rax
  size_t v60; // r14
  size_t v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // [rsp+0h] [rbp-F0h]
  __int64 **v67; // [rsp+8h] [rbp-E8h]
  unsigned int v68; // [rsp+8h] [rbp-E8h]
  __int64 v69; // [rsp+8h] [rbp-E8h]
  void *v70; // [rsp+10h] [rbp-E0h]
  __int64 v71; // [rsp+18h] [rbp-D8h]
  const void *v72; // [rsp+18h] [rbp-D8h]
  unsigned int v73; // [rsp+18h] [rbp-D8h]
  __int64 v74; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v75[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v76[176]; // [rsp+40h] [rbp-B0h] BYREF

  v67 = a2 + 2;
  if ( *((_DWORD *)a2 + 2) == 1 )
    return **a2;
  while ( 1 )
  {
    sub_14637D0((__int64 *)a2, a1[8], a1[7], a3, a4);
    v6 = *a2;
    v7 = **a2;
    v8 = (__int64)*a2;
    if ( !*(_WORD *)(v7 + 24) )
      break;
    v22 = *((_DWORD *)a2 + 2);
    v23 = 0;
    v24 = 0;
    if ( !v22 )
      goto LABEL_31;
LABEL_16:
    v25 = v22 - 1;
    v26 = v23 + 1;
    v27 = v23 + 2 + (unsigned __int64)(v22 - 1 - v23);
    while ( 1 )
    {
      v28 = v24;
      v29 = &v6[v28];
      v30 = v6[v28];
      v31 = *(_WORD *)(v30 + 24);
      if ( v31 > 7u )
        break;
      v24 = v26;
      if ( v27 == v26 + 1 )
        goto LABEL_32;
      ++v26;
    }
    if ( v31 != 8 )
      goto LABEL_32;
    do
    {
      v32 = v8 + 8LL * v22;
      if ( (__int64 *)v32 != v29 + 1 )
      {
        memmove(v29, v29 + 1, v32 - (_QWORD)(v29 + 1));
        v22 = *((_DWORD *)a2 + 2);
      }
      v33 = v22 - 1;
      v34 = *((unsigned int *)a2 + 3);
      *((_DWORD *)a2 + 2) = v33;
      v35 = *(const void **)(v30 + 32);
      v36 = 8LL * *(_QWORD *)(v30 + 40);
      v37 = v36 >> 3;
      if ( v36 >> 3 > (unsigned __int64)(v34 - v33) )
      {
        v72 = v35;
        sub_16CD150(a2, v67, v37 + v33, 8);
        v33 = *((unsigned int *)a2 + 2);
        v35 = v72;
      }
      v8 = (__int64)*a2;
      if ( v36 )
      {
        memcpy((void *)(v8 + 8 * v33), v35, v36);
        v8 = (__int64)*a2;
        LODWORD(v33) = *((_DWORD *)a2 + 2);
      }
      v38 = v33 + v37;
      v29 = (__int64 *)(v8 + v28 * 8);
      *((_DWORD *)a2 + 2) = v38;
      v30 = *(_QWORD *)(v8 + v28 * 8);
      v22 = v38;
    }
    while ( *(_WORD *)(v30 + 24) == 8 );
    if ( v38 == 1 )
      return **a2;
  }
  while ( 1 )
  {
    v9 = v6[1];
    if ( *(_WORD *)(v9 + 24) )
      break;
    v10 = *(_QWORD *)(v9 + 32) + 24LL;
    v11 = *(_QWORD *)(v7 + 32) + 24LL;
    if ( (int)sub_16A9900(v11, v10) <= 0 )
      v11 = v10;
    v12 = sub_15E0530(a1[3]);
    v13 = sub_159C0E0(v12, v11);
    v14 = *a2;
    *v14 = sub_145CE20((__int64)a1, v13);
    v6 = *a2;
    v15 = *a2 + 2;
    v16 = *((unsigned int *)a2 + 2);
    v17 = (__int64)&(*a2)[v16];
    if ( (const void *)v17 != v15 )
    {
      memmove(v6 + 1, v15, v17 - (_QWORD)v15);
      v6 = *a2;
      LODWORD(v16) = *((_DWORD *)a2 + 2);
    }
    v18 = v16 - 1;
    *((_DWORD *)a2 + 2) = v18;
    v7 = *v6;
    if ( v18 == 1 )
      return v7;
  }
  v20 = *(_QWORD *)(v7 + 32);
  v21 = *(_DWORD *)(v20 + 32);
  if ( v21 > 0x40 )
  {
    v71 = v20 + 24;
    if ( v21 != (unsigned int)sub_16A57B0(v20 + 24) )
    {
      if ( v21 == (unsigned int)sub_16A58F0(v71) )
        return v7;
      goto LABEL_13;
    }
LABEL_53:
    v53 = *((unsigned int *)a2 + 2);
    if ( v6 + 1 != &v6[v53] )
    {
      memmove(v6, v6 + 1, 8 * v53 - 8);
      LODWORD(v53) = *((_DWORD *)a2 + 2);
      v6 = *a2;
    }
    v22 = v53 - 1;
    v23 = 0;
    *((_DWORD *)a2 + 2) = v22;
    goto LABEL_14;
  }
  if ( !*(_QWORD *)(v20 + 24) )
    goto LABEL_53;
  if ( *(_QWORD *)(v20 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) )
    return v7;
LABEL_13:
  v22 = *((_DWORD *)a2 + 2);
  v23 = 1;
LABEL_14:
  v8 = (__int64)v6;
  if ( v22 == 1 )
    return *v6;
  v24 = v23;
  if ( v23 < v22 )
    goto LABEL_16;
LABEL_31:
  v25 = v22 - 1;
LABEL_32:
  v73 = v25;
  if ( !v25 )
    return *(_QWORD *)v8;
  v39 = 0;
  while ( 1 )
  {
    v46 = *(_QWORD *)(v8 + 8LL * v39);
    v47 = *(_QWORD *)(v8 + 8LL * (v39 + 1));
    v68 = v39 + 1;
    v48 = 8LL * v39;
    if ( v46 == v47 )
      goto LABEL_35;
    if ( !(unsigned __int8)sub_1481140((__int64)a1, 0x23u, v46, v47) )
      break;
    v8 = (__int64)*a2;
LABEL_35:
    v40 = v8 + v48 + 8;
    v41 = (const void *)(v8 + v48 + 16);
    v42 = 8LL * *((unsigned int *)a2 + 2);
    v43 = (const void *)(v8 + v42);
    v44 = v42 - (v48 + 16);
    if ( v41 != v43 )
    {
      v45 = memmove((void *)(v8 + v48 + 8), v41, v44);
      v8 = (__int64)*a2;
      v40 = (__int64)v45;
    }
    --v73;
    *((_DWORD *)a2 + 2) = (__int64)(v44 + v40 - v8) >> 3;
LABEL_38:
    if ( v73 == v39 )
      goto LABEL_44;
LABEL_39:
    v8 = (__int64)*a2;
  }
  if ( (unsigned __int8)sub_1481140((__int64)a1, 0x25u, (*a2)[v39], (*a2)[v39 + 1]) )
  {
    v54 = (__int64)*a2;
    v55 = v48 + 8;
    v56 = &(*a2)[(unsigned __int64)v48 / 8 + 1];
    v57 = &(*a2)[(unsigned __int64)v48 / 8];
    v58 = *((unsigned int *)a2 + 2);
    v59 = &(*a2)[v58];
    v60 = v58 * 8 - v55;
    if ( v56 != v59 )
    {
      memmove(v57, v56, v60);
      v54 = (__int64)*a2;
    }
    --v73;
    *((_DWORD *)a2 + 2) = (__int64)((__int64)v57 + v60 - v54) >> 3;
    goto LABEL_38;
  }
  ++v39;
  if ( v73 != v68 )
    goto LABEL_39;
LABEL_44:
  if ( *((_DWORD *)a2 + 2) == 1 )
  {
    v8 = (__int64)*a2;
    return *(_QWORD *)v8;
  }
  v49 = 0;
  v75[0] = (unsigned __int64)v76;
  v75[1] = 0x2000000000LL;
  sub_16BD3E0(v75, 8);
  v50 = *((unsigned int *)a2 + 2);
  v51 = 8 * v50;
  if ( (_DWORD)v50 )
  {
    do
    {
      v52 = (*a2)[v49 / 8];
      v49 += 8LL;
      sub_16BD4C0(v75, v52);
    }
    while ( v51 != v49 );
  }
  v74 = 0;
  v7 = sub_16BDDE0(a1 + 102, v75, &v74);
  if ( !v7 )
  {
    v70 = (void *)sub_145CBF0(a1 + 108, 8LL * *((unsigned int *)a2 + 2), 8);
    v61 = 8LL * *((unsigned int *)a2 + 2);
    if ( v61 )
      memmove(v70, *a2, v61);
    v62 = sub_16BD760(v75, a1 + 108);
    v64 = v63;
    v66 = v62;
    v69 = *((unsigned int *)a2 + 2);
    v65 = sub_145CDC0(0x30u, a1 + 108);
    v7 = v65;
    if ( v65 )
    {
      *(_QWORD *)v65 = 0;
      *(_QWORD *)(v65 + 16) = v64;
      *(_QWORD *)(v65 + 8) = v66;
      *(_QWORD *)(v65 + 40) = v69;
      *(_QWORD *)(v65 + 32) = v70;
      *(_DWORD *)(v65 + 24) = 393224;
    }
    sub_16BDA20(a1 + 102, v65, v74);
    sub_146DBF0((__int64)a1, v7);
  }
  if ( (_BYTE *)v75[0] != v76 )
    _libc_free(v75[0]);
  return v7;
}
