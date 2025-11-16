// Function: sub_CB3D40
// Address: 0xcb3d40
//
__int64 __fastcall sub_CB3D40(__int64 a1)
{
  __int64 v2; // rdx
  __int64 *v3; // rbx
  __int64 *v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 *v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rdi
  _QWORD *v12; // r12
  __int64 v13; // rsi
  _QWORD *v14; // r14
  __int64 v15; // rax
  unsigned int v16; // ecx
  __int64 v17; // rdx
  unsigned __int64 v18; // r15
  unsigned __int64 i; // rbx
  __int64 v20; // rdi
  _QWORD *v21; // r14
  _QWORD *v22; // r15
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // rbx
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // rbx
  __int64 *v36; // r14
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rsi
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 *v42; // rbx
  __int64 *v43; // r14
  __int64 v44; // rdi
  unsigned int v45; // ecx
  __int64 v46; // rsi
  __int64 *v47; // rax
  __int64 v48; // rcx
  __int64 *v49; // rbx
  __int64 *v50; // r14
  __int64 v51; // rdi
  unsigned int v52; // ecx
  __int64 v53; // rsi

  v2 = *(unsigned int *)(a1 + 232);
  v3 = *(__int64 **)(a1 + 272);
  v4 = &v3[2 * *(unsigned int *)(a1 + 280)];
  if ( v3 != v4 )
  {
    do
    {
      v5 = v3[1];
      v6 = *v3;
      v3 += 2;
      sub_C7D6A0(v6, v5, 16);
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 232);
  }
  *(_DWORD *)(a1 + 280) = 0;
  if ( (_DWORD)v2 )
  {
    v33 = *(__int64 **)(a1 + 224);
    *(_QWORD *)(a1 + 288) = 0;
    v34 = *v33;
    v35 = &v33[v2];
    v36 = v33 + 1;
    *(_QWORD *)(a1 + 208) = *v33;
    *(_QWORD *)(a1 + 216) = v34 + 4096;
    if ( v35 != v33 + 1 )
    {
      while ( 1 )
      {
        v37 = *v36;
        v38 = (unsigned int)(v36 - v33) >> 7;
        v39 = 4096LL << v38;
        if ( v38 >= 0x1E )
          v39 = 0x40000000000LL;
        ++v36;
        sub_C7D6A0(v37, v39, 16);
        if ( v35 == v36 )
          break;
        v33 = *(__int64 **)(a1 + 224);
      }
    }
    *(_DWORD *)(a1 + 232) = 1;
  }
  v7 = *(__int64 **)(a1 + 368);
  v8 = *(unsigned int *)(a1 + 328);
  v9 = &v7[2 * *(unsigned int *)(a1 + 376)];
  if ( v7 != v9 )
  {
    do
    {
      v10 = v7[1];
      v11 = *v7;
      v7 += 2;
      sub_C7D6A0(v11, v10, 16);
    }
    while ( v9 != v7 );
    v8 = *(unsigned int *)(a1 + 328);
  }
  *(_DWORD *)(a1 + 376) = 0;
  if ( (_DWORD)v8 )
  {
    v47 = *(__int64 **)(a1 + 320);
    *(_QWORD *)(a1 + 384) = 0;
    v48 = *v47;
    v49 = &v47[v8];
    v50 = v47 + 1;
    *(_QWORD *)(a1 + 304) = *v47;
    *(_QWORD *)(a1 + 312) = v48 + 4096;
    if ( v49 != v47 + 1 )
    {
      while ( 1 )
      {
        v51 = *v50;
        v52 = (unsigned int)(v50 - v47) >> 7;
        v53 = 4096LL << v52;
        if ( v52 >= 0x1E )
          v53 = 0x40000000000LL;
        ++v50;
        sub_C7D6A0(v51, v53, 16);
        if ( v49 == v50 )
          break;
        v47 = *(__int64 **)(a1 + 320);
      }
    }
    *(_DWORD *)(a1 + 328) = 1;
  }
  v12 = *(_QWORD **)(a1 + 512);
  v13 = *(unsigned int *)(a1 + 520);
  v14 = &v12[v13];
  if ( v12 != v14 )
  {
    v15 = *(_QWORD *)(a1 + 512);
    while ( 1 )
    {
      v16 = (unsigned int)(((__int64)v12 - v15) >> 3) >> 7;
      v17 = 4096LL << v16;
      if ( v16 >= 0x1E )
        v17 = 0x40000000000LL;
      v18 = *v12 + v17;
      if ( *v12 == *(_QWORD *)(v15 + 8 * v13 - 8) )
        v18 = *(_QWORD *)(a1 + 496);
      for ( i = ((*v12 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 32; i <= v18; i += 32LL )
      {
        v20 = *(_QWORD *)(i - 24);
        if ( v20 )
          j_j___libc_free_0(v20, *(_QWORD *)(i - 8) - v20);
      }
      if ( v14 == ++v12 )
        break;
      v15 = *(_QWORD *)(a1 + 512);
      v13 = *(unsigned int *)(a1 + 520);
    }
  }
  v21 = *(_QWORD **)(a1 + 560);
  v22 = &v21[2 * *(unsigned int *)(a1 + 568)];
  if ( v21 != v22 )
  {
    do
    {
      v23 = *v21 + v21[1];
      v24 = (*v21 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v24 += 32LL;
        if ( v23 < v24 )
          break;
        while ( 1 )
        {
          v25 = *(_QWORD *)(v24 - 24);
          if ( !v25 )
            break;
          v26 = *(_QWORD *)(v24 - 8);
          v24 += 32LL;
          j_j___libc_free_0(v25, v26 - v25);
          if ( v23 < v24 )
            goto LABEL_26;
        }
      }
LABEL_26:
      v21 += 2;
    }
    while ( v22 != v21 );
    v27 = *(__int64 **)(a1 + 560);
    v28 = &v27[2 * *(unsigned int *)(a1 + 568)];
    while ( v28 != v27 )
    {
      v29 = v27[1];
      v30 = *v27;
      v27 += 2;
      sub_C7D6A0(v30, v29, 16);
    }
  }
  v31 = *(unsigned int *)(a1 + 520);
  *(_DWORD *)(a1 + 568) = 0;
  if ( (_DWORD)v31 )
  {
    v40 = *(__int64 **)(a1 + 512);
    *(_QWORD *)(a1 + 576) = 0;
    v41 = *v40;
    v42 = &v40[v31];
    v43 = v40 + 1;
    *(_QWORD *)(a1 + 496) = *v40;
    *(_QWORD *)(a1 + 504) = v41 + 4096;
    if ( v42 != v40 + 1 )
    {
      while ( 1 )
      {
        v44 = *v43;
        v45 = (unsigned int)(v43 - v40) >> 7;
        v46 = 4096LL << v45;
        if ( v45 >= 0x1E )
          v46 = 0x40000000000LL;
        ++v43;
        sub_C7D6A0(v44, v46, 16);
        if ( v42 == v43 )
          break;
        v40 = *(__int64 **)(a1 + 512);
      }
    }
    *(_DWORD *)(a1 + 520) = 1;
  }
  return sub_CB3120(a1 + 400);
}
