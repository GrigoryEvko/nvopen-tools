// Function: sub_9E0DB0
// Address: 0x9e0db0
//
_QWORD *__fastcall sub_9E0DB0(
        __int64 a1,
        int a2,
        const void *a3,
        size_t a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v11; // r14d
  unsigned __int64 v12; // rbx
  __int64 v13; // r14
  bool v14; // zf
  _QWORD *v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r13
  _QWORD *v18; // r12
  unsigned int v19; // esi
  unsigned __int64 v20; // rbx
  __int64 v21; // r8
  unsigned int v22; // edi
  _DWORD *v23; // rax
  int v24; // ecx
  unsigned __int64 *v25; // rax
  _QWORD *v26; // rdi
  _QWORD *result; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r14
  _BYTE *v36; // rdi
  _BYTE *v37; // rax
  size_t v38; // rdx
  __int64 v39; // rax
  int v40; // r14d
  _DWORD *v41; // rdx
  int v42; // eax
  int v43; // ecx
  int v44; // eax
  int v45; // eax
  __int64 v46; // r8
  unsigned int v47; // esi
  int v48; // edi
  int v49; // r10d
  _DWORD *v50; // r9
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdi
  _DWORD *v54; // r8
  unsigned int v55; // r12d
  int v56; // r9d
  int v57; // esi
  __int64 v58; // [rsp+8h] [rbp-B8h]
  size_t v59; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 v62; // [rsp+28h] [rbp-98h]
  unsigned __int64 v63; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v64[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v65[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v66; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v67; // [rsp+70h] [rbp-50h]
  _QWORD *v68; // [rsp+78h] [rbp-48h]
  __int64 v69; // [rsp+80h] [rbp-40h]

  v11 = a5 - 7;
  sub_B2F7A0(v64, a3, a4, a5, a7, a8);
  v62 = sub_B2F650(v64[0], v64[1]);
  v12 = v62;
  if ( v11 <= 1 )
    v62 = sub_B2F650(a3, a4);
  if ( (_BYTE)qword_4F80468 )
  {
    v28 = sub_C5F790();
    v29 = *(_QWORD *)(v28 + 32);
    v30 = v28;
    if ( (unsigned __int64)(*(_QWORD *)(v28 + 24) - v29) <= 4 )
    {
      v30 = sub_CB6200(v28, "GUID ", 5);
    }
    else
    {
      *(_DWORD *)v29 = 1145656647;
      *(_BYTE *)(v29 + 4) = 32;
      *(_QWORD *)(v28 + 32) += 5LL;
    }
    v31 = sub_CB59D0(v30, v12);
    v32 = *(_BYTE **)(v31 + 32);
    if ( *(_BYTE **)(v31 + 24) == v32 )
    {
      v31 = sub_CB6200(v31, "(", 1);
    }
    else
    {
      *v32 = 40;
      ++*(_QWORD *)(v31 + 32);
    }
    v33 = sub_CB59D0(v31, v62);
    v34 = *(_QWORD *)(v33 + 32);
    v35 = v33;
    if ( (unsigned __int64)(*(_QWORD *)(v33 + 24) - v34) <= 4 )
    {
      v39 = sub_CB6200(v33, ") is ", 5);
      v36 = *(_BYTE **)(v39 + 32);
      v35 = v39;
    }
    else
    {
      *(_DWORD *)v34 = 1936269353;
      *(_BYTE *)(v34 + 4) = 32;
      v36 = (_BYTE *)(*(_QWORD *)(v33 + 32) + 5LL);
      *(_QWORD *)(v33 + 32) = v36;
    }
    v37 = *(_BYTE **)(v35 + 24);
    if ( v37 - v36 < a4 )
    {
      v35 = sub_CB6200(v35, a3, a4);
      v36 = *(_BYTE **)(v35 + 32);
      if ( v36 != *(_BYTE **)(v35 + 24) )
      {
LABEL_32:
        *v36 = 10;
        ++*(_QWORD *)(v35 + 32);
        goto LABEL_4;
      }
    }
    else
    {
      if ( a4 )
      {
        memcpy(v36, a3, a4);
        v37 = *(_BYTE **)(v35 + 24);
        v36 = (_BYTE *)(a4 + *(_QWORD *)(v35 + 32));
        *(_QWORD *)(v35 + 32) = v36;
      }
      if ( v36 != v37 )
        goto LABEL_32;
    }
    sub_CB6200(v35, "\n", 1);
  }
LABEL_4:
  v13 = *(_QWORD *)(a1 + 424);
  if ( *(_BYTE *)(a1 + 384) )
  {
    v58 = (__int64)a3;
    v59 = a4;
  }
  else
  {
    v58 = sub_C948A0(v13 + 512, a3, a4);
    v59 = v38;
  }
  v14 = *(_BYTE *)(v13 + 343) == 0;
  v63 = v12;
  if ( v14 )
  {
    v66.m128i_i64[1] = 0;
    v66.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v66.m128i_i64[0] = 0;
  }
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v15 = sub_9CA390((_QWORD *)v13, &v63, &v66);
  v16 = v68;
  v17 = v67;
  v18 = v15;
  v60 = (unsigned __int64)(v15 + 4);
  if ( v68 != v67 )
  {
    do
    {
      if ( *v17 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v17 + 8LL))(*v17);
      ++v17;
    }
    while ( v16 != v17 );
    v17 = v67;
  }
  if ( v17 )
    j_j___libc_free_0(v17, v69 - (_QWORD)v17);
  v18[5] = v58;
  v18[6] = v59;
  v19 = *(_DWORD *)(a1 + 472);
  v20 = *(unsigned __int8 *)(v13 + 343) | v60 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 448);
    goto LABEL_49;
  }
  v21 = *(_QWORD *)(a1 + 456);
  v22 = (v19 - 1) & (37 * a2);
  v23 = (_DWORD *)(v21 + 24LL * v22);
  v24 = *v23;
  if ( a2 == *v23 )
  {
LABEL_17:
    v25 = (unsigned __int64 *)(v23 + 2);
    goto LABEL_18;
  }
  v40 = 1;
  v41 = 0;
  while ( v24 != -1 )
  {
    if ( !v41 && v24 == -2 )
      v41 = v23;
    v22 = (v19 - 1) & (v40 + v22);
    v23 = (_DWORD *)(v21 + 24LL * v22);
    v24 = *v23;
    if ( a2 == *v23 )
      goto LABEL_17;
    ++v40;
  }
  if ( !v41 )
    v41 = v23;
  v42 = *(_DWORD *)(a1 + 464);
  ++*(_QWORD *)(a1 + 448);
  v43 = v42 + 1;
  if ( 4 * (v42 + 1) >= 3 * v19 )
  {
LABEL_49:
    sub_9E0BD0(a1 + 448, 2 * v19);
    v44 = *(_DWORD *)(a1 + 472);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 456);
      v43 = *(_DWORD *)(a1 + 464) + 1;
      v47 = v45 & (37 * a2);
      v41 = (_DWORD *)(v46 + 24LL * v47);
      v48 = *v41;
      if ( a2 != *v41 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -1 )
        {
          if ( v48 == -2 && !v50 )
            v50 = v41;
          v47 = v45 & (v49 + v47);
          v41 = (_DWORD *)(v46 + 24LL * v47);
          v48 = *v41;
          if ( a2 == *v41 )
            goto LABEL_45;
          ++v49;
        }
        if ( v50 )
          v41 = v50;
      }
      goto LABEL_45;
    }
    goto LABEL_77;
  }
  if ( v19 - *(_DWORD *)(a1 + 468) - v43 <= v19 >> 3 )
  {
    sub_9E0BD0(a1 + 448, v19);
    v51 = *(_DWORD *)(a1 + 472);
    if ( v51 )
    {
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 456);
      v54 = 0;
      v55 = v52 & (37 * a2);
      v56 = 1;
      v43 = *(_DWORD *)(a1 + 464) + 1;
      v41 = (_DWORD *)(v53 + 24LL * v55);
      v57 = *v41;
      if ( a2 != *v41 )
      {
        while ( v57 != -1 )
        {
          if ( !v54 && v57 == -2 )
            v54 = v41;
          v55 = v52 & (v56 + v55);
          v41 = (_DWORD *)(v53 + 24LL * v55);
          v57 = *v41;
          if ( a2 == *v41 )
            goto LABEL_45;
          ++v56;
        }
        if ( v54 )
          v41 = v54;
      }
      goto LABEL_45;
    }
LABEL_77:
    ++*(_DWORD *)(a1 + 464);
    BUG();
  }
LABEL_45:
  *(_DWORD *)(a1 + 464) = v43;
  if ( *v41 != -1 )
    --*(_DWORD *)(a1 + 468);
  *((_QWORD *)v41 + 1) = 0;
  *((_QWORD *)v41 + 2) = 0;
  *v41 = a2;
  v25 = (unsigned __int64 *)(v41 + 2);
LABEL_18:
  *v25 = v20;
  v26 = (_QWORD *)v64[0];
  v25[1] = v62;
  result = v65;
  if ( v26 != v65 )
    return (_QWORD *)j_j___libc_free_0(v26, v65[0] + 1LL);
  return result;
}
