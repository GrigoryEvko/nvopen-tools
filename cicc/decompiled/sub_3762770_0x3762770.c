// Function: sub_3762770
// Address: 0x3762770
//
__int64 __fastcall sub_3762770(__int64 a1, unsigned __int64 a2, unsigned __int16 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // r12d
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // r15
  unsigned int v12; // ecx
  const __m128i *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // r13d
  char v17; // al
  __int64 v18; // r8
  int v19; // esi
  unsigned int v20; // edx
  _DWORD *v21; // rdi
  int v22; // r10d
  _DWORD *v23; // r13
  unsigned int v24; // esi
  unsigned int v25; // edx
  _DWORD *v26; // rcx
  int v27; // edi
  unsigned int v28; // r10d
  int v29; // r11d
  int v30; // eax
  __int64 v31; // rdi
  int v32; // eax
  unsigned int v33; // esi
  int v34; // edx
  int v35; // eax
  __int64 v36; // rdi
  int v37; // eax
  unsigned int v38; // edx
  int v39; // esi
  int v40; // r10d
  _DWORD *v41; // r8
  int v42; // r10d
  __m128i v43; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE *v44; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+28h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = *(unsigned int *)(a2 + 24);
  v6 = *(_QWORD *)a1;
  if ( (unsigned int)v5 <= 0x1F3 && (!a3 || *(_BYTE *)(v5 + v6 + 500LL * a3 + 6414) != 4) )
    return 0;
  v9 = *(_QWORD *)(a1 + 8);
  v44 = v46;
  v45 = 0x800000000LL;
  (*(void (__fastcall **)(__int64, unsigned __int64, _BYTE **, __int64))(*(_QWORD *)v6 + 2424LL))(v6, a2, &v44, v9);
  v10 = (unsigned int)v45;
  if ( !(_DWORD)v45 )
  {
    v7 = 0;
    goto LABEL_18;
  }
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v12 = v11;
      v13 = (const __m128i *)&v44[16 * v11];
      v14 = *(_QWORD *)(a2 + 48) + 16 * v11;
      v15 = *(_QWORD *)(v13->m128i_i64[0] + 48) + 16LL * v13->m128i_u32[2];
      if ( *(_WORD *)v14 != *(_WORD *)v15 || *(_QWORD *)(v14 + 8) != *(_QWORD *)(v15 + 8) && !*(_WORD *)v14 )
        break;
      ++v11;
      sub_3760E70(a1, a2, v12, v13->m128i_i64[0], v13->m128i_i64[1]);
      if ( v10 == v11 )
        goto LABEL_17;
    }
    v43 = _mm_loadu_si128(v13);
    sub_375EAB0(a1, (__int64)&v43);
    v16 = sub_375D5B0(a1, a2, (unsigned int)v11);
    v17 = *(_BYTE *)(a1 + 1456) & 1;
    if ( v17 )
    {
      v18 = a1 + 1464;
      v19 = 7;
    }
    else
    {
      v24 = *(_DWORD *)(a1 + 1472);
      v18 = *(_QWORD *)(a1 + 1464);
      if ( !v24 )
      {
        v25 = *(_DWORD *)(a1 + 1456);
        ++*(_QWORD *)(a1 + 1448);
        v26 = 0;
        v27 = (v25 >> 1) + 1;
LABEL_23:
        v28 = 3 * v24;
        goto LABEL_24;
      }
      v19 = v24 - 1;
    }
    v20 = v19 & (37 * v16);
    v21 = (_DWORD *)(v18 + 8LL * v20);
    v22 = *v21;
    if ( v16 == *v21 )
    {
LABEL_15:
      v23 = v21 + 1;
      goto LABEL_16;
    }
    v29 = 1;
    v26 = 0;
    while ( v22 != -1 )
    {
      if ( v22 == -2 && !v26 )
        v26 = v21;
      v20 = v19 & (v29 + v20);
      v21 = (_DWORD *)(v18 + 8LL * v20);
      v22 = *v21;
      if ( v16 == *v21 )
        goto LABEL_15;
      ++v29;
    }
    v25 = *(_DWORD *)(a1 + 1456);
    v28 = 24;
    v24 = 8;
    if ( !v26 )
      v26 = v21;
    ++*(_QWORD *)(a1 + 1448);
    v27 = (v25 >> 1) + 1;
    if ( !v17 )
    {
      v24 = *(_DWORD *)(a1 + 1472);
      goto LABEL_23;
    }
LABEL_24:
    if ( 4 * v27 >= v28 )
    {
      sub_375BDE0(a1 + 1448, 2 * v24);
      if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
      {
        v31 = a1 + 1464;
        v32 = 7;
      }
      else
      {
        v30 = *(_DWORD *)(a1 + 1472);
        v31 = *(_QWORD *)(a1 + 1464);
        if ( !v30 )
          goto LABEL_68;
        v32 = v30 - 1;
      }
      v33 = v32 & (37 * v16);
      v26 = (_DWORD *)(v31 + 8LL * v33);
      v34 = *v26;
      if ( v16 == *v26 )
        goto LABEL_40;
      v42 = 1;
      v41 = 0;
      while ( v34 != -1 )
      {
        if ( v34 == -2 && !v41 )
          v41 = v26;
        v33 = v32 & (v42 + v33);
        v26 = (_DWORD *)(v31 + 8LL * v33);
        v34 = *v26;
        if ( v16 == *v26 )
          goto LABEL_40;
        ++v42;
      }
LABEL_47:
      if ( v41 )
        v26 = v41;
LABEL_40:
      v25 = *(_DWORD *)(a1 + 1456);
      goto LABEL_26;
    }
    if ( v24 - *(_DWORD *)(a1 + 1460) - v27 <= v24 >> 3 )
    {
      sub_375BDE0(a1 + 1448, v24);
      if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
      {
        v36 = a1 + 1464;
        v37 = 7;
      }
      else
      {
        v35 = *(_DWORD *)(a1 + 1472);
        v36 = *(_QWORD *)(a1 + 1464);
        if ( !v35 )
        {
LABEL_68:
          *(_DWORD *)(a1 + 1456) = (2 * (*(_DWORD *)(a1 + 1456) >> 1) + 2) | *(_DWORD *)(a1 + 1456) & 1;
          BUG();
        }
        v37 = v35 - 1;
      }
      v38 = v37 & (37 * v16);
      v26 = (_DWORD *)(v36 + 8LL * v38);
      v39 = *v26;
      if ( v16 == *v26 )
        goto LABEL_40;
      v40 = 1;
      v41 = 0;
      while ( v39 != -1 )
      {
        if ( v39 == -2 && !v41 )
          v41 = v26;
        v38 = v37 & (v40 + v38);
        v26 = (_DWORD *)(v36 + 8LL * v38);
        v39 = *v26;
        if ( v16 == *v26 )
          goto LABEL_40;
        ++v40;
      }
      goto LABEL_47;
    }
LABEL_26:
    *(_DWORD *)(a1 + 1456) = (2 * (v25 >> 1) + 2) | v25 & 1;
    if ( *v26 != -1 )
      --*(_DWORD *)(a1 + 1460);
    *v26 = v16;
    v23 = v26 + 1;
    v26[1] = 0;
LABEL_16:
    ++v11;
    *v23 = sub_375D5B0(a1, v43.m128i_u64[0], v43.m128i_i64[1]);
  }
  while ( v10 != v11 );
LABEL_17:
  v7 = 1;
LABEL_18:
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  return v7;
}
