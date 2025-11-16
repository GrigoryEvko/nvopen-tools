// Function: sub_1CF2F90
// Address: 0x1cf2f90
//
__m128i *__fastcall sub_1CF2F90(__m128i *a1, __int64 a2, __int64 a3, const __m128i *a4, __int64 a5)
{
  __m128i v8; // xmm0
  bool v9; // zf
  __int64 **v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // r14
  unsigned int v17; // esi
  __int32 v18; // r15d
  __int64 v19; // r8
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  unsigned __int64 v26; // rax
  int v27; // r11d
  __int64 *v28; // rdx
  int v29; // eax
  int v30; // ecx
  int v31; // eax
  int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // r8
  int v36; // r10d
  __int64 *v37; // r9
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdi
  __int64 *v41; // r8
  unsigned int v42; // r14d
  int v43; // r9d
  __int64 v44; // rsi
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 **v46; // [rsp+18h] [rbp-68h] BYREF
  __int64 v47[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v48[2]; // [rsp+30h] [rbp-50h] BYREF
  char v49; // [rsp+40h] [rbp-40h]
  char v50; // [rsp+41h] [rbp-3Fh]

  v8 = _mm_loadu_si128(a4);
  v9 = *(_BYTE *)(a5 + 8) == 0;
  v10 = *(__int64 ***)a3;
  a1[1].m128i_i64[0] = a4[1].m128i_i64[0];
  v11 = a4->m128i_i64[0];
  *a1 = v8;
  if ( v9 )
  {
    v45 = v11;
    v26 = sub_157EBA0(*(_QWORD *)(v11 + 40));
    v11 = v45;
    v12 = v26;
  }
  else
  {
    v12 = *(_QWORD *)a5;
  }
  v47[0] = v11;
  v50 = 1;
  v48[0] = (__int64)"pcp";
  v49 = 3;
  v46 = v10;
  v47[1] = sub_1599EF0(v10);
  v13 = sub_1CF0110(4042, (__int64 *)&v46, 1, v47, 2, (__int64)v48, v12);
  v14 = *(_QWORD *)(a3 + 48);
  a1->m128i_i64[0] = v13;
  v15 = v13;
  v48[0] = v14;
  if ( v14 )
  {
    v16 = v13 + 48;
    sub_1623A60((__int64)v48, v14, 2);
    if ( (__int64 *)(v15 + 48) == v48 )
    {
      if ( v48[0] )
        sub_161E7C0(v15 + 48, v48[0]);
      goto LABEL_7;
    }
  }
  else
  {
    v16 = v13 + 48;
    if ( (__int64 *)(v13 + 48) == v48 )
      goto LABEL_7;
  }
  v24 = *(_QWORD *)(v15 + 48);
  if ( v24 )
    sub_161E7C0(v16, v24);
  v25 = (unsigned __int8 *)v48[0];
  *(_QWORD *)(v15 + 48) = v48[0];
  if ( v25 )
    sub_1623210((__int64)v48, v25, v16);
LABEL_7:
  v17 = *(_DWORD *)(a2 + 112);
  v18 = a1[1].m128i_i32[0];
  if ( !v17 )
  {
    ++*(_QWORD *)(a2 + 88);
    goto LABEL_26;
  }
  v19 = *(_QWORD *)(a2 + 96);
  v20 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v21 = (__int64 *)(v19 + 16LL * v20);
  v22 = *v21;
  if ( v15 == *v21 )
    return a1;
  v27 = 1;
  v28 = 0;
  while ( v22 != -8 )
  {
    if ( v22 != -16 || v28 )
      v21 = v28;
    v20 = (v17 - 1) & (v27 + v20);
    v22 = *(_QWORD *)(v19 + 16LL * v20);
    if ( v15 == v22 )
      return a1;
    ++v27;
    v28 = v21;
    v21 = (__int64 *)(v19 + 16LL * v20);
  }
  if ( !v28 )
    v28 = v21;
  v29 = *(_DWORD *)(a2 + 104);
  ++*(_QWORD *)(a2 + 88);
  v30 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v17 )
  {
LABEL_26:
    sub_1541C50(a2 + 88, 2 * v17);
    v31 = *(_DWORD *)(a2 + 112);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a2 + 96);
      v34 = (v31 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v30 = *(_DWORD *)(a2 + 104) + 1;
      v28 = (__int64 *)(v33 + 16LL * v34);
      v35 = *v28;
      if ( v15 != *v28 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -8 )
        {
          if ( v35 == -16 && !v37 )
            v37 = v28;
          v34 = v32 & (v36 + v34);
          v28 = (__int64 *)(v33 + 16LL * v34);
          v35 = *v28;
          if ( v15 == *v28 )
            goto LABEL_22;
          ++v36;
        }
        if ( v37 )
          v28 = v37;
      }
      goto LABEL_22;
    }
    goto LABEL_54;
  }
  if ( v17 - *(_DWORD *)(a2 + 108) - v30 <= v17 >> 3 )
  {
    sub_1541C50(a2 + 88, v17);
    v38 = *(_DWORD *)(a2 + 112);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a2 + 96);
      v41 = 0;
      v42 = v39 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v43 = 1;
      v30 = *(_DWORD *)(a2 + 104) + 1;
      v28 = (__int64 *)(v40 + 16LL * v42);
      v44 = *v28;
      if ( v15 != *v28 )
      {
        while ( v44 != -8 )
        {
          if ( !v41 && v44 == -16 )
            v41 = v28;
          v42 = v39 & (v43 + v42);
          v28 = (__int64 *)(v40 + 16LL * v42);
          v44 = *v28;
          if ( v15 == *v28 )
            goto LABEL_22;
          ++v43;
        }
        if ( v41 )
          v28 = v41;
      }
      goto LABEL_22;
    }
LABEL_54:
    ++*(_DWORD *)(a2 + 104);
    BUG();
  }
LABEL_22:
  *(_DWORD *)(a2 + 104) = v30;
  if ( *v28 != -8 )
    --*(_DWORD *)(a2 + 108);
  *v28 = v15;
  *((_DWORD *)v28 + 2) = v18;
  return a1;
}
