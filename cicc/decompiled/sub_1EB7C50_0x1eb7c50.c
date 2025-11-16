// Function: sub_1EB7C50
// Address: 0x1eb7c50
//
__int64 __fastcall sub_1EB7C50(__int64 a1, __int64 *a2, __m128i *a3)
{
  __m128i *v4; // r12
  __int64 v6; // rax
  __int64 *v7; // rbx
  bool v8; // zf
  int v9; // esi
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  unsigned int v12; // esi
  __int32 v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // edx
  int v16; // ecx
  __int64 v17; // r8
  __int64 *v18; // r14
  __int64 *v19; // r12
  __int64 *v20; // rbx
  __int64 v21; // rdx
  int v22; // r10d
  _DWORD *v23; // r9
  int v24; // eax
  int v25; // edx
  int v26; // eax
  __int32 v27; // esi
  int v28; // eax
  __int64 v29; // r8
  __int64 v30; // rcx
  int v31; // edi
  _DWORD *v32; // r10
  int v33; // r11d
  _DWORD *v34; // r9
  int v35; // eax
  __int32 v36; // esi
  int v37; // ecx
  __int64 v38; // r8
  __int64 v39; // rdi
  int v40; // eax
  int v41; // r11d
  __m128i *v42; // [rsp+8h] [rbp-48h]
  __int64 *v43; // [rsp+10h] [rbp-40h]
  bool v44; // [rsp+18h] [rbp-38h]
  _DWORD *v45; // [rsp+18h] [rbp-38h]

  v4 = a3;
  if ( !a3[1].m128i_i8[0] )
    return sub_1EB6BB0(a1, v4);
  v6 = a3->m128i_i64[0];
  v7 = a2;
  a3[1].m128i_i8[0] = 0;
  v8 = v6 == (_QWORD)a2;
  v9 = a3->m128i_i32[2];
  v43 = (__int64 *)v6;
  v44 = !v8;
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 24LL) + 16LL * (v9 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (unsigned int)sub_1EB6550((__int64 *)a1, v9, v10);
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64 *, _QWORD, bool, __int64, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 408LL))(
    *(_QWORD *)(a1 + 256),
    *(_QWORD *)(a1 + 360),
    v7,
    v4->m128i_u16[6],
    v44,
    v11,
    v10,
    *(_QWORD *)(a1 + 248));
  v12 = *(_DWORD *)(a1 + 640);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 616);
    goto LABEL_21;
  }
  v13 = v4->m128i_i32[2];
  v14 = *(_QWORD *)(a1 + 624);
  v15 = (v12 - 1) & (37 * v13);
  v45 = (_DWORD *)(v14 + 56LL * v15);
  v16 = *v45;
  if ( v13 != *v45 )
  {
    v22 = 1;
    v23 = 0;
    while ( v16 != -1 )
    {
      if ( !v23 && v16 == -2 )
        v23 = v45;
      v15 = (v12 - 1) & (v22 + v15);
      v45 = (_DWORD *)(v14 + 56LL * v15);
      v16 = *v45;
      if ( v13 == *v45 )
        goto LABEL_5;
      ++v22;
    }
    v24 = *(_DWORD *)(a1 + 632);
    if ( !v23 )
      v23 = v45;
    ++*(_QWORD *)(a1 + 616);
    v25 = v24 + 1;
    v45 = v23;
    if ( 4 * (v24 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 636) - v25 > v12 >> 3 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 632) = v25;
        if ( *v45 != -1 )
          --*(_DWORD *)(a1 + 636);
        *v45 = v4->m128i_i32[2];
        *((_QWORD *)v45 + 1) = v45 + 6;
        *((_QWORD *)v45 + 2) = 0x400000000LL;
        goto LABEL_9;
      }
      sub_1EB7930(a1 + 616, v12);
      v35 = *(_DWORD *)(a1 + 640);
      if ( v35 )
      {
        v36 = v4->m128i_i32[2];
        v37 = v35 - 1;
        v38 = *(_QWORD *)(a1 + 624);
        LODWORD(v39) = (v35 - 1) & (37 * v36);
        v45 = (_DWORD *)(v38 + 56LL * (unsigned int)v39);
        v40 = *v45;
        v25 = *(_DWORD *)(a1 + 632) + 1;
        if ( *v45 == v36 )
          goto LABEL_17;
        v32 = (_DWORD *)(v38 + 56LL * (unsigned int)v39);
        v41 = 1;
        v34 = 0;
        while ( v40 != -1 )
        {
          if ( !v34 && v40 == -2 )
            v34 = v32;
          v39 = v37 & (unsigned int)(v39 + v41);
          v32 = (_DWORD *)(v38 + 56 * v39);
          v40 = *v32;
          if ( v36 == *v32 )
            goto LABEL_36;
          ++v41;
        }
        goto LABEL_25;
      }
      goto LABEL_48;
    }
LABEL_21:
    sub_1EB7930(a1 + 616, 2 * v12);
    v26 = *(_DWORD *)(a1 + 640);
    if ( v26 )
    {
      v27 = v4->m128i_i32[2];
      v28 = v26 - 1;
      v29 = *(_QWORD *)(a1 + 624);
      LODWORD(v30) = v28 & (37 * v27);
      v45 = (_DWORD *)(v29 + 56LL * (unsigned int)v30);
      v31 = *v45;
      v25 = *(_DWORD *)(a1 + 632) + 1;
      if ( *v45 == v27 )
        goto LABEL_17;
      v32 = (_DWORD *)(v29 + 56LL * (v28 & (unsigned int)(37 * v27)));
      v33 = 1;
      v34 = 0;
      while ( v31 != -1 )
      {
        if ( !v34 && v31 == -2 )
          v34 = v32;
        v30 = v28 & (unsigned int)(v30 + v33);
        v32 = (_DWORD *)(v29 + 56 * v30);
        v31 = *v32;
        if ( v27 == *v32 )
        {
LABEL_36:
          v45 = v32;
          goto LABEL_17;
        }
        ++v33;
      }
LABEL_25:
      if ( !v34 )
        v34 = v32;
      v45 = v34;
      goto LABEL_17;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 632);
    BUG();
  }
LABEL_5:
  v17 = *((_QWORD *)v45 + 1);
  if ( v17 + 8LL * (unsigned int)v45[4] != v17 )
  {
    v42 = v4;
    v18 = (__int64 *)*((_QWORD *)v45 + 1);
    v19 = v7;
    v20 = (__int64 *)(v17 + 8LL * (unsigned int)v45[4]);
    do
    {
      v21 = *v18++;
      sub_1E1BF70(*(_QWORD *)(a1 + 360), v19, v21, v11);
    }
    while ( v20 != v18 );
    v7 = v19;
    v4 = v42;
  }
LABEL_9:
  v45[4] = 0;
  if ( v43 != v7 )
    v4->m128i_i64[0] = 0;
  return sub_1EB6BB0(a1, v4);
}
