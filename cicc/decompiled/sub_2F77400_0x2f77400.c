// Function: sub_2F77400
// Address: 0x2f77400
//
void __fastcall sub_2F77400(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r12
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int *v10; // r14
  unsigned int *v11; // r12
  unsigned __int64 v12; // rsi
  unsigned int v13; // edi
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  _BYTE *v22; // r8
  signed __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r12
  _BYTE *v29; // rbx
  unsigned int v30; // esi
  unsigned int v31; // r8d
  __int64 v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // r11
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  int v40; // edx
  unsigned __int64 v41; // rcx
  __int64 v42; // rdi
  unsigned __int64 v43; // rax
  __int64 i; // rsi
  __int16 v45; // dx
  __int64 v46; // rsi
  unsigned int v47; // edi
  unsigned int v48; // ecx
  __int64 *v49; // rdx
  __int64 v50; // r9
  int v51; // edx
  int v52; // r11d
  _BYTE *v53; // [rsp+10h] [rbp-2A0h] BYREF
  __int64 v54; // [rsp+18h] [rbp-298h]
  _BYTE v55[192]; // [rsp+20h] [rbp-290h] BYREF
  unsigned int *v56; // [rsp+E0h] [rbp-1D0h]
  __int64 v57; // [rsp+E8h] [rbp-1C8h]
  _BYTE v58[192]; // [rsp+F0h] [rbp-1C0h] BYREF
  unsigned int *v59; // [rsp+1B0h] [rbp-100h]
  __int64 v60; // [rsp+1B8h] [rbp-F8h]
  _BYTE v61[240]; // [rsp+1C0h] [rbp-F0h] BYREF

  v3 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v40 = *(_DWORD *)(a2 + 44);
    v41 = a2;
    v42 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    v43 = a2;
    if ( (v40 & 4) != 0 )
    {
      do
        v43 = *(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v43 + 44) & 4) != 0 );
    }
    if ( (v40 & 8) != 0 )
    {
      do
        v41 = *(_QWORD *)(v41 + 8);
      while ( (*(_BYTE *)(v41 + 44) & 8) != 0 );
    }
    for ( i = *(_QWORD *)(v41 + 8); i != v43; v43 = *(_QWORD *)(v43 + 8) )
    {
      v45 = *(_WORD *)(v43 + 68);
      if ( (unsigned __int16)(v45 - 14) > 4u && v45 != 24 )
        break;
    }
    v46 = *(_QWORD *)(v42 + 128);
    v47 = *(_DWORD *)(v42 + 144);
    if ( v47 )
    {
      v48 = (v47 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v49 = (__int64 *)(v46 + 16LL * v48);
      v50 = *v49;
      if ( *v49 == v43 )
      {
LABEL_68:
        v3 = v49[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_2;
      }
      v51 = 1;
      while ( v50 != -4096 )
      {
        v52 = v51 + 1;
        v48 = (v47 - 1) & (v51 + v48);
        v49 = (__int64 *)(v46 + 16LL * v48);
        v50 = *v49;
        if ( *v49 == v43 )
          goto LABEL_68;
        v51 = v52;
      }
    }
    v49 = (__int64 *)(v46 + 16LL * v47);
    goto LABEL_68;
  }
LABEL_2:
  v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(unsigned __int8 *)(a1 + 58);
  v53 = v55;
  v56 = (unsigned int *)v58;
  v54 = 0x800000000LL;
  v57 = 0x800000000LL;
  v59 = (unsigned int *)v61;
  v60 = 0x800000000LL;
  sub_2F75980((__int64)&v53, a2, v6, v5, v7, 1);
  if ( *(_BYTE *)(a1 + 58) )
  {
    sub_2F76630((__int64 *)&v53, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v3, 0);
  }
  else if ( *(_BYTE *)(a1 + 56) )
  {
    sub_2F761E0((__int64)&v53, a2, *(_QWORD *)(a1 + 32), v8, v9);
  }
  sub_2F77060(a1, v59, (unsigned int)v60);
  v10 = v56;
  v11 = &v56[6 * (unsigned int)v57];
  if ( v11 != v56 )
  {
    while ( 1 )
    {
      v12 = *v10;
      v13 = v12;
      if ( (v12 & 0x80000000) != 0LL )
        v13 = *(_DWORD *)(a1 + 320) + (v12 & 0x7FFFFFFF);
      v14 = *(unsigned int *)(a1 + 104);
      v15 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v13);
      if ( v15 >= (unsigned int)v14 )
        goto LABEL_42;
      v16 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v17 = v16 + 24LL * v15;
        if ( v13 == *(_DWORD *)v17 )
          break;
        v15 += 256;
        if ( (unsigned int)v14 <= v15 )
          goto LABEL_42;
      }
      if ( v17 == v16 + 24 * v14 )
      {
LABEL_42:
        v19 = 0;
        v18 = 0;
      }
      else
      {
        v18 = *(_QWORD *)(v17 + 8);
        v19 = *(_QWORD *)(v17 + 16);
      }
      v20 = (__int64)v53;
      v21 = 24LL * (unsigned int)v54;
      v22 = &v53[v21];
      v23 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
      if ( v23 >> 2 )
      {
        while ( (_DWORD)v12 != *(_DWORD *)v20 )
        {
          if ( (_DWORD)v12 == *(_DWORD *)(v20 + 24) )
          {
            v20 += 24;
            goto LABEL_21;
          }
          if ( (_DWORD)v12 == *(_DWORD *)(v20 + 48) )
          {
            v20 += 48;
            goto LABEL_21;
          }
          if ( (_DWORD)v12 == *(_DWORD *)(v20 + 72) )
          {
            v20 += 72;
            goto LABEL_21;
          }
          v20 += 96;
          if ( &v53[96 * (v23 >> 2)] == (_BYTE *)v20 )
          {
            v23 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v22[-v20] >> 3);
            goto LABEL_44;
          }
        }
        goto LABEL_21;
      }
LABEL_44:
      if ( v23 == 2 )
        goto LABEL_50;
      if ( v23 == 3 )
        break;
      if ( v23 != 1 )
      {
LABEL_47:
        v25 = 0;
        v24 = 0;
        goto LABEL_23;
      }
LABEL_52:
      if ( (_DWORD)v12 != *(_DWORD *)v20 )
      {
        v25 = 0;
        v24 = 0;
        goto LABEL_23;
      }
LABEL_21:
      if ( v22 == (_BYTE *)v20 )
        goto LABEL_47;
      v24 = *(_QWORD *)(v20 + 8);
      v25 = *(_QWORD *)(v20 + 16);
LABEL_23:
      v26 = *((_QWORD *)v10 + 2);
      v27 = *((_QWORD *)v10 + 1);
      v10 += 6;
      sub_2F74F40(a1, v12, v18, v19, v18 & (v24 | v18 & ~v27), v19 & (v25 | v19 & ~v26));
      if ( v11 == v10 )
        goto LABEL_24;
    }
    if ( (_DWORD)v12 == *(_DWORD *)v20 )
      goto LABEL_21;
    v20 += 24;
LABEL_50:
    if ( (_DWORD)v12 == *(_DWORD *)v20 )
      goto LABEL_21;
    v20 += 24;
    goto LABEL_52;
  }
LABEL_24:
  v28 = (__int64)v53;
  v29 = &v53[24 * (unsigned int)v54];
  if ( v29 != v53 )
  {
    do
    {
      v30 = *(_DWORD *)v28;
      v31 = *(_DWORD *)v28;
      if ( *(int *)v28 < 0 )
        v31 = *(_DWORD *)(a1 + 320) + (v31 & 0x7FFFFFFF);
      v32 = *(unsigned int *)(a1 + 104);
      v33 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v31);
      if ( v33 >= (unsigned int)v32 )
        goto LABEL_41;
      v34 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v35 = v34 + 24LL * v33;
        if ( v31 == *(_DWORD *)v35 )
          break;
        v33 += 256;
        if ( (unsigned int)v32 <= v33 )
          goto LABEL_41;
      }
      if ( v35 == v34 + 24 * v32 )
      {
LABEL_41:
        v37 = 0;
        v36 = 0;
      }
      else
      {
        v36 = *(_QWORD *)(v35 + 8);
        v37 = *(_QWORD *)(v35 + 16);
      }
      v38 = *(_QWORD *)(v28 + 8);
      v39 = *(_QWORD *)(v28 + 16);
      v28 += 24;
      sub_2F74DB0(a1, v30, v36, v37, v36 | v38, v37 | v39);
    }
    while ( v29 != (_BYTE *)v28 );
  }
  if ( v59 != (unsigned int *)v61 )
    _libc_free((unsigned __int64)v59);
  if ( v56 != (unsigned int *)v58 )
    _libc_free((unsigned __int64)v56);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
}
