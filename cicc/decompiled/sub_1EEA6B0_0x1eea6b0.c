// Function: sub_1EEA6B0
// Address: 0x1eea6b0
//
__int64 __fastcall sub_1EEA6B0(unsigned int *a1, unsigned int a2, __int64 a3, unsigned int a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v8; // rdx
  __int64 v9; // r11
  int v10; // r10d
  int v11; // r13d
  __int64 v12; // rax
  unsigned int v13; // esi
  int v14; // r12d
  unsigned int v15; // r8d
  unsigned int v16; // r9d
  int *v17; // rax
  unsigned int v18; // edx
  unsigned int v19; // r15d
  int v20; // ecx
  __int64 v21; // rcx
  __int64 v22; // rdi
  unsigned int v23; // ecx
  unsigned int v24; // ecx
  __int64 v25; // r15
  _QWORD *v26; // rdi
  __int64 (*v27)(); // rax
  int v28; // r9d
  unsigned __int64 v29; // rsi
  __int64 i; // rsi
  _BYTE *v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rsi
  __int64 j; // rsi
  _BYTE *v35; // rdx
  __int64 v36; // rcx
  unsigned int v38; // eax
  __int64 v39; // r8
  _QWORD *v40; // rax
  int *v41; // rax
  const char *v42; // r13
  const char *v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rcx
  const char *v46; // rsi
  _DWORD *v47; // rdi
  int *v51; // [rsp+18h] [rbp-E8h]
  unsigned int v52; // [rsp+20h] [rbp-E0h]
  __m128i v54[2]; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD v55[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v56; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v57[2]; // [rsp+70h] [rbp-90h] BYREF
  __m128i v58[2]; // [rsp+90h] [rbp-70h] BYREF
  __m128i v59[5]; // [rsp+B0h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(sub_1E15F70((__int64)a5) + 56);
  v9 = *(_QWORD *)(v8 + 8);
  v10 = *(_DWORD *)(v8 + 32);
  v11 = -v10;
  v12 = *(_QWORD *)(*(_QWORD *)a1 + 280LL)
      + 24LL
      * (*(unsigned __int16 *)(*(_QWORD *)a3 + 24LL)
       + *(_DWORD *)(*(_QWORD *)a1 + 288LL)
       * (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 264LL) - *(_QWORD *)(*(_QWORD *)a1 + 256LL)) >> 3));
  v13 = a1[14];
  v14 = -858993459 * ((*(_QWORD *)(v8 + 16) - v9) >> 3) - v10;
  v15 = *(_DWORD *)(v12 + 4) >> 3;
  v16 = *(_DWORD *)(v12 + 8) >> 3;
  if ( !v13 )
  {
    v38 = 0;
    v25 = 0;
LABEL_36:
    v39 = (unsigned int)v14;
    if ( a1[15] <= v38 )
    {
      sub_16CD150((__int64)(a1 + 12), a1 + 16, 0, 16, v14, v16);
      v13 = a1[14];
      v39 = (unsigned int)v14;
    }
    v40 = (_QWORD *)(*((_QWORD *)a1 + 6) + 16LL * v13);
    *v40 = v39;
    v40[1] = 0;
    v41 = (int *)*((_QWORD *)a1 + 6);
    ++a1[14];
    v51 = v41;
    goto LABEL_12;
  }
  v17 = (int *)*((_QWORD *)a1 + 6);
  v52 = a1[14];
  v18 = 0;
  v19 = -1;
  v51 = v17;
  do
  {
    if ( !v17[1] )
    {
      v20 = *v17;
      if ( *v17 >= v11 && v20 < v14 )
      {
        v21 = v9 + 40LL * (unsigned int)(v10 + v20);
        v22 = *(_QWORD *)(v21 + 8);
        v23 = *(_DWORD *)(v21 + 16);
        if ( v15 <= (unsigned int)v22 && v23 >= v16 )
        {
          v24 = v22 + v23 - (v16 + v15);
          if ( v24 < v19 )
          {
            v52 = v18;
            v19 = v24;
          }
        }
      }
    }
    ++v18;
    v17 += 4;
  }
  while ( v18 != v13 );
  v25 = 16LL * v52;
  if ( v52 == v13 )
  {
    v38 = v13;
    goto LABEL_36;
  }
LABEL_12:
  v51[(unsigned __int64)v25 / 4 + 1] = a2;
  v26 = *(_QWORD **)a1;
  v27 = *(__int64 (**)())(**(_QWORD **)a1 + 384LL);
  if ( v27 != sub_1EE9930 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD *, _QWORD, _QWORD *, _QWORD *, __int64, _QWORD))v27)(
           v26,
           *((_QWORD *)a1 + 3),
           a5,
           a6,
           a3,
           a2) )
    {
      return v25 + *((_QWORD *)a1 + 6);
    }
    v26 = *(_QWORD **)a1;
  }
  v28 = *(_DWORD *)(*((_QWORD *)a1 + 6) + v25);
  if ( v28 < v11 || v28 >= v14 )
  {
    v42 = (const char *)(v26[10] + *(unsigned int *)(*(_QWORD *)a3 + 16LL));
    v43 = (const char *)(v26[9] + *(unsigned int *)(v26[1] + 24LL * a2));
    v55[0] = &v56;
    v59[0].m128i_i64[0] = 28;
    v44 = sub_22409D0(v55, v59, 0);
    v45 = 7;
    v46 = "Error while trying to spill ";
    v55[0] = v44;
    v47 = (_DWORD *)v44;
    v56 = v59[0].m128i_i64[0];
    while ( v45 )
    {
      *v47 = *(_DWORD *)v46;
      v46 += 4;
      ++v47;
      --v45;
    }
    v55[1] = v59[0].m128i_i64[0];
    *(_BYTE *)(v55[0] + v59[0].m128i_i64[0]) = 0;
    sub_94F930(v57, (__int64)v55, v43);
    sub_94F930(v58, (__int64)v57, " from class ");
    sub_94F930(v59, (__int64)v58, v42);
    sub_94F930(v54, (__int64)v59, ": Cannot scavenge register without an emergency spill slot!");
    sub_2240A30(v59);
    sub_2240A30(v58);
    sub_2240A30(v57);
    sub_2240A30(v55);
    sub_16BD130(v54[0].m128i_i64[0], 1u);
  }
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD *, _QWORD, __int64))(**((_QWORD **)a1 + 1) + 408LL))(
    *((_QWORD *)a1 + 1),
    *((_QWORD *)a1 + 3),
    a5,
    a2,
    1);
  v29 = *a5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v29 )
    goto LABEL_44;
  if ( (*(_QWORD *)v29 & 4) == 0 && (*(_BYTE *)(v29 + 46) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v29; ; i = *(_QWORD *)v29 )
    {
      v29 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v29 + 46) & 4) == 0 )
        break;
    }
  }
  v31 = *(_BYTE **)(v29 + 32);
  v32 = 0;
  if ( *v31 != 5 )
  {
    do
      v32 = (unsigned int)(v32 + 1);
    while ( v31[40 * v32] != 5 );
  }
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, __int64, unsigned int *))(**(_QWORD **)a1 + 392LL))(
    *(_QWORD *)a1,
    v29,
    a4,
    v32,
    a1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD))(**((_QWORD **)a1 + 1) + 416LL))(
    *((_QWORD *)a1 + 1),
    *((_QWORD *)a1 + 3),
    *a6,
    a2,
    *(unsigned int *)(*((_QWORD *)a1 + 6) + v25),
    a3,
    *(_QWORD *)a1);
  v33 = *(_QWORD *)*a6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v33 )
LABEL_44:
    BUG();
  if ( (*(_QWORD *)v33 & 4) == 0 && (*(_BYTE *)(v33 + 46) & 4) != 0 )
  {
    for ( j = *(_QWORD *)v33; ; j = *(_QWORD *)v33 )
    {
      v33 = j & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v33 + 46) & 4) == 0 )
        break;
    }
  }
  v35 = *(_BYTE **)(v33 + 32);
  v36 = 0;
  if ( *v35 != 5 )
  {
    do
      v36 = (unsigned int)(v36 + 1);
    while ( v35[40 * v36] != 5 );
  }
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, __int64, unsigned int *))(**(_QWORD **)a1 + 392LL))(
    *(_QWORD *)a1,
    v33,
    a4,
    v36,
    a1);
  return v25 + *((_QWORD *)a1 + 6);
}
