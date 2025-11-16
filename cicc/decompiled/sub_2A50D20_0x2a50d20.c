// Function: sub_2A50D20
// Address: 0x2a50d20
//
void __fastcall sub_2A50D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rbx
  __int64 *v17; // r13
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  _BYTE *v22; // r13
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r14
  __int64 v29; // r12
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rdx
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // rax
  __m128i *v38; // rsi
  __int64 *v39; // r13
  __int64 v40; // r14
  __int64 v41; // r15
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r8
  __int64 v46; // rdx
  const void *v47; // [rsp+28h] [rbp-1C8h]
  __int64 v48; // [rsp+38h] [rbp-1B8h]
  __int64 *v49; // [rsp+38h] [rbp-1B8h]
  __int64 *v50; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v51; // [rsp+48h] [rbp-1A8h]
  _BYTE v52[16]; // [rsp+50h] [rbp-1A0h] BYREF
  _BYTE v53[32]; // [rsp+60h] [rbp-190h] BYREF
  __m128i v54; // [rsp+80h] [rbp-170h] BYREF
  char v55; // [rsp+90h] [rbp-160h]
  _BYTE *v56; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-138h]
  _BYTE v58[48]; // [rsp+C0h] [rbp-130h] BYREF
  __m128i v59; // [rsp+F0h] [rbp-100h] BYREF
  char v60; // [rsp+100h] [rbp-F0h] BYREF
  unsigned __int64 v61[2]; // [rsp+130h] [rbp-C0h] BYREF
  _BYTE v62[88]; // [rsp+140h] [rbp-B0h] BYREF
  int v63; // [rsp+198h] [rbp-58h] BYREF
  unsigned __int64 v64; // [rsp+1A0h] [rbp-50h]
  int *v65; // [rsp+1A8h] [rbp-48h]
  int *v66; // [rsp+1B0h] [rbp-40h]
  __int64 v67; // [rsp+1B8h] [rbp-38h]

  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_BYTE *)(a1 + 560) = 1;
  *(_DWORD *)(a1 + 576) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_DWORD *)(a1 + 624) = 0;
  *(_DWORD *)(a1 + 688) = 0;
  v7 = *(_QWORD *)(a2 + 16);
  if ( v7 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 24);
      v12 = *(_QWORD *)(v11 + 40);
      if ( *(_BYTE *)v11 == 62 )
      {
        v8 = *(unsigned int *)(a1 + 8);
        if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + 1, 8u, a5, a6);
          v8 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v8) = v12;
        ++*(_DWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 544) = v11;
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 280);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 284) )
        {
          sub_C8D5F0(a1 + 272, (const void *)(a1 + 288), v13 + 1, 8u, a5, a6);
          v13 = *(unsigned int *)(a1 + 280);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v13) = v12;
        ++*(_DWORD *)(a1 + 280);
      }
      if ( !*(_BYTE *)(a1 + 560) )
        goto LABEL_10;
      v9 = *(_QWORD *)(a1 + 552);
      v10 = *(_QWORD *)(v11 + 40);
      if ( v9 )
      {
        if ( v9 != v10 )
          *(_BYTE *)(a1 + 560) = 0;
LABEL_10:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
      }
      else
      {
        *(_QWORD *)(a1 + 552) = v10;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
      }
    }
  }
  v50 = (__int64 *)v52;
  v51 = 0x100000000LL;
  v56 = v58;
  v57 = 0x600000000LL;
  sub_AE7A50((__int64)&v50, a2, (__int64)&v56);
  v16 = v50;
  v17 = &v50[(unsigned int)v51];
  if ( v50 != v17 )
  {
    do
    {
      v18 = *v16;
      v19 = *(_QWORD *)(*v16 - 32);
      if ( !v19 || *(_BYTE *)v19 || *(_QWORD *)(v19 + 24) != *(_QWORD *)(v18 + 80) )
        BUG();
      if ( *(_DWORD *)(v19 + 36) != 68 )
      {
        v20 = *(unsigned int *)(a1 + 576);
        if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 580) )
        {
          sub_C8D5F0(a1 + 568, (const void *)(a1 + 584), v20 + 1, 8u, v14, v15);
          v20 = *(unsigned int *)(a1 + 576);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 568) + 8 * v20) = v18;
        ++*(_DWORD *)(a1 + 576);
      }
      ++v16;
    }
    while ( v17 != v16 );
  }
  v21 = (unsigned __int64)v56;
  v22 = &v56[8 * (unsigned int)v57];
  if ( v56 != v22 )
  {
    do
    {
      v23 = *(_QWORD *)v21;
      if ( *(_BYTE *)(*(_QWORD *)v21 + 64LL) != 2 )
      {
        v24 = *(unsigned int *)(a1 + 600);
        if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 604) )
        {
          sub_C8D5F0(a1 + 592, (const void *)(a1 + 608), v24 + 1, 8u, v14, v15);
          v24 = *(unsigned int *)(a1 + 600);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 592) + 8 * v24) = v23;
        ++*(_DWORD *)(a1 + 600);
      }
      v21 += 8LL;
    }
    while ( v22 != (_BYTE *)v21 );
  }
  v63 = 0;
  v61[0] = (unsigned __int64)v62;
  v61[1] = 0x200000000LL;
  v65 = &v63;
  v66 = &v63;
  v64 = 0;
  v67 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v25 = sub_B91C10(a2, 38);
    if ( !v25 || (v26 = sub_AE94B0(v25), v48 = v27, v28 = v26, v26 == v27) )
    {
      v35 = *(_BYTE *)(a2 + 7) & 0x20;
    }
    else
    {
      do
      {
        while ( 1 )
        {
          v29 = *(_QWORD *)(v28 + 24);
          sub_AF4850((__int64)&v59, v29);
          sub_2A50B10((__int64)&v54, (__int64)v61, &v59, v30, v31, v32);
          if ( v55 )
            break;
          v28 = *(_QWORD *)(v28 + 8);
          if ( v28 == v48 )
            goto LABEL_40;
        }
        v34 = *(unsigned int *)(a1 + 624);
        if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 628) )
        {
          sub_C8D5F0(a1 + 616, (const void *)(a1 + 632), v34 + 1, 8u, v33, v34 + 1);
          v34 = *(unsigned int *)(a1 + 624);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 616) + 8 * v34) = v29;
        ++*(_DWORD *)(a1 + 624);
        v28 = *(_QWORD *)(v28 + 8);
      }
      while ( v28 != v48 );
LABEL_40:
      v35 = *(_BYTE *)(a2 + 7) & 0x20;
    }
    if ( v35 )
    {
      v36 = sub_B91C10(a2, 38);
      if ( v36 )
      {
        v37 = *(_QWORD *)(v36 + 8);
        v38 = (__m128i *)(v37 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v37 & 4) == 0 )
          v38 = 0;
        sub_B967C0(&v59, v38);
        v49 = (__int64 *)(v59.m128i_i64[0] + 8LL * v59.m128i_u32[2]);
        if ( (__int64 *)v59.m128i_i64[0] != v49 )
        {
          v39 = (__int64 *)v59.m128i_i64[0];
          v40 = a1;
          v47 = (const void *)(a1 + 696);
          do
          {
            while ( 1 )
            {
              v41 = *v39;
              sub_AF48C0(&v54, *v39);
              sub_2A50B10((__int64)v53, (__int64)v61, &v54, v42, v43, v44);
              if ( v53[16] )
                break;
              if ( v49 == ++v39 )
                goto LABEL_52;
            }
            v46 = *(unsigned int *)(v40 + 688);
            if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 692) )
            {
              sub_C8D5F0(v40 + 680, v47, v46 + 1, 8u, v45, v46 + 1);
              v46 = *(unsigned int *)(v40 + 688);
            }
            ++v39;
            *(_QWORD *)(*(_QWORD *)(v40 + 680) + 8 * v46) = v41;
            ++*(_DWORD *)(v40 + 688);
          }
          while ( v49 != v39 );
LABEL_52:
          v49 = (__int64 *)v59.m128i_i64[0];
        }
        if ( v49 != (__int64 *)&v60 )
          _libc_free((unsigned __int64)v49);
      }
    }
  }
  sub_2A4CC40(v64);
  if ( (_BYTE *)v61[0] != v62 )
    _libc_free(v61[0]);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( v50 != (__int64 *)v52 )
    _libc_free((unsigned __int64)v50);
}
