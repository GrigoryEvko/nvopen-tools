// Function: sub_D8C2B0
// Address: 0xd8c2b0
//
_QWORD *__fastcall sub_D8C2B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 i; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __m128i *v17; // r14
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // r13
  __int64 v21; // rbx
  _QWORD *v22; // r12
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rcx
  _QWORD *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rdi
  __int64 *v28; // rdi
  __int64 v29; // r13
  __int64 v30; // r14
  __int64 v31; // rbx
  int v32; // r13d
  __int64 v33; // r15
  __int64 *v34; // r13
  __int64 v35; // rdx
  unsigned int v36; // esi
  unsigned int v37; // ecx
  __int64 v38; // rax
  _BOOL8 v39; // rdi
  _BOOL8 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // [rsp+0h] [rbp-7E0h]
  __m128i **v50; // [rsp+10h] [rbp-7D0h]
  _QWORD *v51; // [rsp+20h] [rbp-7C0h]
  __m128i *v52; // [rsp+28h] [rbp-7B8h]
  __int64 v53; // [rsp+28h] [rbp-7B8h]
  __int64 v55; // [rsp+30h] [rbp-7B0h]
  __int64 v56; // [rsp+30h] [rbp-7B0h]
  __int64 v57; // [rsp+30h] [rbp-7B0h]
  __int64 v58; // [rsp+30h] [rbp-7B0h]
  __int64 v59; // [rsp+30h] [rbp-7B0h]
  __m128i **v60; // [rsp+38h] [rbp-7A8h]
  __int64 v61; // [rsp+38h] [rbp-7A8h]
  __m128i **v62; // [rsp+40h] [rbp-7A0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-798h]
  _BYTE v64[512]; // [rsp+50h] [rbp-790h] BYREF
  _BYTE v65[1424]; // [rsp+250h] [rbp-590h] BYREF

  v7 = (_QWORD *)a1;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  v51 = (_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 72) = a1 + 56;
  *(_QWORD *)(a1 + 80) = a1 + 56;
  v49 = a1 + 56;
  v62 = (__m128i **)v64;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  v63 = 0x4000000000LL;
  v8 = *(_QWORD *)(*a2 + 80);
  v9 = *a2 + 72;
  if ( v9 == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v8 + 32);
      if ( i != v8 + 24 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 == v8 )
        break;
      if ( !v8 )
        BUG();
    }
  }
LABEL_7:
  while ( v9 != v8 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 60 )
    {
      v11 = (unsigned int)v63;
      v12 = (unsigned int)v63 + 1LL;
      if ( v12 > HIDWORD(v63) )
      {
        sub_C8D5F0((__int64)&v62, v64, v12, 8u, a5, a6);
        v11 = (unsigned int)v63;
      }
      v62[v11] = (__m128i *)(i - 24);
      LODWORD(v63) = v63 + 1;
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v8 + 32) )
    {
      v13 = v8 - 24;
      if ( !v8 )
        v13 = 0;
      if ( i != v13 + 48 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
  v14 = (__m128i *)*a2;
  sub_1054520(v65, *a2, v62, (unsigned int)v63, 1);
  sub_1051870(v65);
  v60 = v62;
  v50 = &v62[(unsigned int)v63];
  if ( v50 == v62 )
    goto LABEL_35;
  do
  {
    v17 = *v60;
    v52 = *v60;
    v18 = sub_22077B0(168);
    v19 = *((_DWORD *)a2 + 6);
    *(_QWORD *)(v18 + 32) = v17;
    v20 = v18 + 40;
    v21 = v18;
    sub_AADB10(v18 + 40, v19, 0);
    *(_DWORD *)(v21 + 80) = 0;
    *(_QWORD *)(v21 + 96) = v21 + 80;
    *(_QWORD *)(v21 + 104) = v21 + 80;
    *(_QWORD *)(v21 + 144) = v21 + 128;
    *(_QWORD *)(v21 + 152) = v21 + 128;
    *(_QWORD *)(v21 + 88) = 0;
    v22 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(v21 + 112) = 0;
    *(_DWORD *)(v21 + 128) = 0;
    *(_QWORD *)(v21 + 136) = 0;
    *(_QWORD *)(v21 + 160) = 0;
    if ( !v22 )
    {
      if ( v51 == *(_QWORD **)(a1 + 24) )
      {
        v22 = v51;
        v41 = 1;
        goto LABEL_59;
      }
      v22 = v51;
LABEL_61:
      v42 = sub_220EF80(v22);
      if ( *(_QWORD *)(v42 + 32) >= *(_QWORD *)(v21 + 32) )
      {
        v22 = (_QWORD *)v42;
        goto LABEL_30;
      }
LABEL_57:
      v41 = 1;
      if ( v51 != v22 )
        v41 = *(_QWORD *)(v21 + 32) < v22[4];
LABEL_59:
      sub_220F040(v41, v21, v22, v51);
      ++*(_QWORD *)(a1 + 40);
      goto LABEL_33;
    }
    v23 = *(_QWORD *)(v21 + 32);
    while ( 1 )
    {
      v24 = v22[4];
      v25 = (_QWORD *)v22[3];
      if ( v23 < v24 )
        v25 = (_QWORD *)v22[2];
      if ( !v25 )
        break;
      v22 = v25;
    }
    if ( v23 < v24 )
    {
      if ( *(_QWORD **)(a1 + 24) == v22 )
        goto LABEL_57;
      goto LABEL_61;
    }
    if ( v23 > v24 )
      goto LABEL_57;
LABEL_30:
    sub_D85C20(0);
    v26 = *(_QWORD *)(v21 + 88);
    while ( v26 )
    {
      sub_D85A50(*(_QWORD *)(v26 + 24));
      v27 = v26;
      v26 = *(_QWORD *)(v26 + 16);
      j_j___libc_free_0(v27, 40);
    }
    sub_969240((__int64 *)(v21 + 56));
    v28 = (__int64 *)v20;
    v20 = (__int64)(v22 + 5);
    sub_969240(v28);
    j_j___libc_free_0(v21, 168);
LABEL_33:
    v14 = v52;
    sub_D8B020((__int64)a2, v52, v20, (__int64)v65);
    ++v60;
  }
  while ( v50 != v60 );
  v7 = (_QWORD *)a1;
LABEL_35:
  v29 = *a2;
  if ( (*(_BYTE *)(*a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*a2, (__int64)v14, v15, v16);
    v30 = *(_QWORD *)(v29 + 96);
    v31 = v30 + 40LL * *(_QWORD *)(v29 + 104);
    if ( (*(_BYTE *)(v29 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v29, (__int64)v14, v43, v44);
      v30 = *(_QWORD *)(v29 + 96);
    }
  }
  else
  {
    v30 = *(_QWORD *)(v29 + 96);
    v31 = v30 + 40LL * *(_QWORD *)(v29 + 104);
  }
  if ( v31 != v30 )
  {
    v61 = (__int64)a2;
    while ( 1 )
    {
      while ( *(_BYTE *)(*(_QWORD *)(v30 + 8) + 8LL) != 14 || (unsigned __int8)sub_B2D680(v30) )
      {
        v30 += 40;
        if ( v30 == v31 )
          goto LABEL_54;
      }
      v32 = *(_DWORD *)(v30 + 32);
      v33 = sub_22077B0(168);
      *(_DWORD *)(v33 + 32) = v32;
      v34 = (__int64 *)(v33 + 40);
      sub_AADB10(v33 + 40, *(_DWORD *)(v61 + 24), 0);
      v35 = v7[8];
      *(_DWORD *)(v33 + 80) = 0;
      *(_QWORD *)(v33 + 96) = v33 + 80;
      *(_QWORD *)(v33 + 104) = v33 + 80;
      *(_QWORD *)(v33 + 88) = 0;
      *(_QWORD *)(v33 + 112) = 0;
      *(_DWORD *)(v33 + 128) = 0;
      *(_QWORD *)(v33 + 136) = 0;
      *(_QWORD *)(v33 + 144) = v33 + 128;
      *(_QWORD *)(v33 + 152) = v33 + 128;
      *(_QWORD *)(v33 + 160) = 0;
      if ( v35 )
        break;
      if ( v7[9] != v49 )
      {
        v35 = v49;
        goto LABEL_69;
      }
      v35 = v49;
      v39 = 1;
LABEL_52:
      sub_220F040(v39, v33, v35, v49);
      ++v7[11];
LABEL_53:
      v14 = (__m128i *)v30;
      v30 += 40;
      sub_D8B020(v61, v14, (__int64)v34, (__int64)v65);
      if ( v30 == v31 )
        goto LABEL_54;
    }
    v36 = *(_DWORD *)(v33 + 32);
    while ( 1 )
    {
      v37 = *(_DWORD *)(v35 + 32);
      v38 = *(_QWORD *)(v35 + 24);
      if ( v36 < v37 )
        v38 = *(_QWORD *)(v35 + 16);
      if ( !v38 )
        break;
      v35 = v38;
    }
    if ( v36 < v37 )
    {
      if ( v35 != v7[9] )
      {
LABEL_69:
        v55 = v35;
        v45 = sub_220EF80(v35);
        v35 = v55;
        if ( *(_DWORD *)(v45 + 32) >= *(_DWORD *)(v33 + 32) )
        {
          v35 = v45;
LABEL_71:
          v56 = v35;
          sub_D85C20(0);
          v46 = *(_QWORD *)(v33 + 88);
          v47 = v56;
          if ( v46 )
          {
            do
            {
              v53 = v47;
              v57 = v46;
              sub_D85A50(*(_QWORD *)(v46 + 24));
              v48 = v57;
              v58 = *(_QWORD *)(v57 + 16);
              j_j___libc_free_0(v48, 40);
              v46 = v58;
              v47 = v53;
            }
            while ( v58 );
          }
          v59 = v47;
          sub_969240((__int64 *)(v33 + 56));
          sub_969240(v34);
          j_j___libc_free_0(v33, 168);
          v34 = (__int64 *)(v59 + 40);
          goto LABEL_53;
        }
      }
    }
    else if ( v36 <= v37 )
    {
      goto LABEL_71;
    }
    v39 = 1;
    if ( v49 != v35 )
      v39 = *(_DWORD *)(v33 + 32) < *(_DWORD *)(v35 + 32);
    goto LABEL_52;
  }
LABEL_54:
  sub_D896C0((__int64)v65, (__int64)v14);
  if ( v62 != (__m128i **)v64 )
    _libc_free(v62, v14);
  return v7;
}
