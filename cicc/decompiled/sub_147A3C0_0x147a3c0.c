// Function: sub_147A3C0
// Address: 0x147a3c0
//
__int64 __fastcall sub_147A3C0(_QWORD *a1, __int64 **a2, __m128i a3, __m128i a4)
{
  __int64 *v6; // r12
  __int64 v7; // r15
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rbx
  const void *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  unsigned int v23; // ebx
  __int64 v24; // rdi
  unsigned int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // rbx
  unsigned int v28; // r10d
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 *v32; // rdi
  __int64 v33; // r15
  unsigned __int16 v34; // dx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rax
  const void *v38; // r10
  signed __int64 v39; // r15
  __int64 v40; // r12
  unsigned int v41; // r12d
  unsigned int v42; // ebx
  unsigned int v43; // r12d
  __int64 v44; // rdx
  void *v45; // rcx
  const void *v46; // rsi
  __int64 v47; // r15
  const void *v48; // rax
  size_t v49; // r15
  void *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r15
  unsigned __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // rbx
  __int64 v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 *v61; // rsi
  __int64 *v62; // r15
  __int64 v63; // r8
  __int64 *v64; // rax
  size_t v65; // r8
  void *v66; // rbx
  size_t v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r12
  __int64 v71; // rax
  __int64 v72; // [rsp+0h] [rbp-F0h]
  __int64 v73; // [rsp+8h] [rbp-E8h]
  __int64 **v74; // [rsp+10h] [rbp-E0h]
  unsigned int v75; // [rsp+10h] [rbp-E0h]
  const void *v76; // [rsp+18h] [rbp-D8h]
  size_t v77; // [rsp+18h] [rbp-D8h]
  __int64 v78; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v79[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v80[176]; // [rsp+40h] [rbp-B0h] BYREF

  v74 = a2 + 2;
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
    v25 = *((_DWORD *)a2 + 2);
    v26 = 0;
    v27 = 0;
    if ( !v25 )
      goto LABEL_31;
LABEL_16:
    v28 = v25 - 1;
    v29 = v26 + 1;
    v30 = v26 + 2 + (unsigned __int64)(v25 - 1 - v26);
    while ( 1 )
    {
      v31 = v27;
      v32 = &v6[v31];
      v33 = v6[v31];
      v34 = *(_WORD *)(v33 + 24);
      if ( v34 > 8u )
        break;
      v27 = v29;
      if ( v30 == v29 + 1 )
        goto LABEL_32;
      ++v29;
    }
    if ( v34 != 9 )
      goto LABEL_32;
    do
    {
      v35 = v8 + 8LL * v25;
      if ( (__int64 *)v35 != v32 + 1 )
      {
        memmove(v32, v32 + 1, v35 - (_QWORD)(v32 + 1));
        v25 = *((_DWORD *)a2 + 2);
      }
      v36 = v25 - 1;
      v37 = *((unsigned int *)a2 + 3);
      *((_DWORD *)a2 + 2) = v36;
      v38 = *(const void **)(v33 + 32);
      v39 = 8LL * *(_QWORD *)(v33 + 40);
      v40 = v39 >> 3;
      if ( v39 >> 3 > (unsigned __int64)(v37 - v36) )
      {
        v76 = v38;
        sub_16CD150(a2, v74, v40 + v36, 8);
        v36 = *((unsigned int *)a2 + 2);
        v38 = v76;
      }
      v8 = (__int64)*a2;
      if ( v39 )
      {
        memcpy((void *)(v8 + 8 * v36), v38, v39);
        v8 = (__int64)*a2;
        LODWORD(v36) = *((_DWORD *)a2 + 2);
      }
      v41 = v36 + v40;
      v32 = (__int64 *)(v8 + v31 * 8);
      *((_DWORD *)a2 + 2) = v41;
      v33 = *(_QWORD *)(v8 + v31 * 8);
      v25 = v41;
    }
    while ( *(_WORD *)(v33 + 24) == 9 );
    if ( v41 == 1 )
      return **a2;
  }
  while ( 1 )
  {
    v9 = v6[1];
    if ( *(_WORD *)(v9 + 24) )
      break;
    v10 = *(_QWORD *)(v9 + 32) + 24LL;
    v11 = *(_QWORD *)(v7 + 32) + 24LL;
    if ( (int)sub_16AEA10(v11, v10) <= 0 )
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
  v22 = *(_QWORD *)(v20 + 24);
  v23 = v21 - 1;
  if ( v21 <= 0x40 )
  {
    if ( 1LL << v23 == v22 )
      goto LABEL_51;
    if ( (1LL << v23) - 1 == v22 )
      return v7;
  }
  else
  {
    v24 = v20 + 24;
    if ( (*(_QWORD *)(v22 + 8LL * (v23 >> 6)) & (1LL << v23)) == 0 )
    {
      if ( (unsigned int)sub_16A58F0(v24) == v23 )
        return v7;
      goto LABEL_13;
    }
    if ( (unsigned int)sub_16A58A0(v24) == v23 )
    {
LABEL_51:
      v58 = *((unsigned int *)a2 + 2);
      if ( v6 + 1 != &v6[v58] )
      {
        memmove(v6, v6 + 1, 8 * v58 - 8);
        LODWORD(v58) = *((_DWORD *)a2 + 2);
        v6 = *a2;
      }
      v25 = v58 - 1;
      v26 = 0;
      *((_DWORD *)a2 + 2) = v25;
      goto LABEL_14;
    }
  }
LABEL_13:
  v25 = *((_DWORD *)a2 + 2);
  v26 = 1;
LABEL_14:
  v8 = (__int64)v6;
  if ( v25 == 1 )
    return *v6;
  v27 = v26;
  if ( v26 < v25 )
    goto LABEL_16;
LABEL_31:
  v28 = v25 - 1;
LABEL_32:
  v42 = v28;
  if ( !v28 )
    return *(_QWORD *)v8;
  v43 = 0;
  while ( 1 )
  {
    v51 = *(_QWORD *)(v8 + 8LL * v43);
    v52 = *(_QWORD *)(v8 + 8LL * (v43 + 1));
    v53 = 8LL * v43;
    v75 = v43 + 1;
    if ( v51 == v52 )
      goto LABEL_35;
    if ( !(unsigned __int8)sub_147A340((__int64)a1, 0x27u, v51, v52) )
      break;
    v8 = (__int64)*a2;
LABEL_35:
    v44 = v53 + 16;
    v45 = (void *)(v8 + v53 + 8);
    v46 = (const void *)(v8 + v53 + 16);
    v47 = 8LL * *((unsigned int *)a2 + 2);
    v48 = (const void *)(v8 + v47);
    v49 = v47 - v44;
    if ( v46 != v48 )
    {
      v50 = memmove(v45, v46, v49);
      v8 = (__int64)*a2;
      v45 = v50;
    }
    --v42;
    *((_DWORD *)a2 + 2) = ((__int64)((__int64)v45 + v49) - v8) >> 3;
LABEL_38:
    if ( v42 == v43 )
      goto LABEL_44;
LABEL_39:
    v8 = (__int64)*a2;
  }
  if ( (unsigned __int8)sub_147A340((__int64)a1, 0x29u, (*a2)[v43], (*a2)[v43 + 1]) )
  {
    v59 = (__int64)*a2;
    v60 = v53 + 8;
    v61 = &(*a2)[(unsigned __int64)v53 / 8 + 1];
    v62 = &(*a2)[(unsigned __int64)v53 / 8];
    v63 = *((unsigned int *)a2 + 2);
    v64 = &(*a2)[v63];
    v65 = v63 * 8 - v60;
    if ( v61 != v64 )
    {
      v77 = v65;
      memmove(v62, v61, v65);
      v59 = (__int64)*a2;
      v65 = v77;
    }
    --v42;
    *((_DWORD *)a2 + 2) = (__int64)((__int64)v62 + v65 - v59) >> 3;
    goto LABEL_38;
  }
  ++v43;
  if ( v42 != v75 )
    goto LABEL_39;
LABEL_44:
  if ( *((_DWORD *)a2 + 2) == 1 )
  {
    v8 = (__int64)*a2;
    return *(_QWORD *)v8;
  }
  v54 = 0;
  v79[0] = (unsigned __int64)v80;
  v79[1] = 0x2000000000LL;
  sub_16BD3E0(v79, 9);
  v55 = *((unsigned int *)a2 + 2);
  v56 = 8 * v55;
  if ( (_DWORD)v55 )
  {
    do
    {
      v57 = (*a2)[v54 / 8];
      v54 += 8LL;
      sub_16BD4C0(v79, v57);
    }
    while ( v56 != v54 );
  }
  v78 = 0;
  v7 = sub_16BDDE0(a1 + 102, v79, &v78);
  if ( !v7 )
  {
    v66 = (void *)sub_145CBF0(a1 + 108, 8LL * *((unsigned int *)a2 + 2), 8);
    v67 = 8LL * *((unsigned int *)a2 + 2);
    if ( v67 )
      memmove(v66, *a2, v67);
    v68 = sub_16BD760(v79, a1 + 108);
    v70 = v69;
    v72 = v68;
    v73 = *((unsigned int *)a2 + 2);
    v71 = sub_145CDC0(0x30u, a1 + 108);
    v7 = v71;
    if ( v71 )
    {
      *(_QWORD *)v71 = 0;
      *(_QWORD *)(v71 + 16) = v70;
      *(_QWORD *)(v71 + 8) = v72;
      *(_QWORD *)(v71 + 32) = v66;
      *(_QWORD *)(v71 + 40) = v73;
      *(_DWORD *)(v71 + 24) = 393225;
    }
    sub_16BDA20(a1 + 102, v71, v78);
    sub_146DBF0((__int64)a1, v7);
  }
  if ( (_BYTE *)v79[0] != v80 )
    _libc_free(v79[0]);
  return v7;
}
