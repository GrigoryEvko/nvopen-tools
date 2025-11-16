// Function: sub_2965A20
// Address: 0x2965a20
//
__int64 __fastcall sub_2965A20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  char v6; // di
  __int64 v7; // r11
  int v8; // r9d
  unsigned int v9; // edx
  __int64 *v10; // r10
  __int64 v11; // rbx
  __int64 v12; // rax
  char v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r11
  int v16; // eax
  unsigned int v17; // edx
  __int64 *v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v22; // rdx
  __int64 v23; // r10
  _QWORD *v24; // r9
  __int64 v25; // rbx
  unsigned int v26; // r13d
  __int64 v27; // rdi
  int v28; // edx
  bool v29; // of
  int v30; // r10d
  __int64 v31; // rdi
  int v32; // edi
  unsigned int v33; // eax
  __int64 v34; // rsi
  int v35; // eax
  unsigned int v36; // r8d
  unsigned int v37; // edx
  int v38; // ecx
  int v39; // ecx
  unsigned int v40; // esi
  __int64 v41; // [rsp+8h] [rbp-78h]
  _QWORD *v42; // [rsp+10h] [rbp-70h]
  _QWORD *v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+20h] [rbp-60h] BYREF
  __int64 v45; // [rsp+28h] [rbp-58h] BYREF
  __int64 v46; // [rsp+30h] [rbp-50h] BYREF
  __m128i v47; // [rsp+38h] [rbp-48h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_BYTE *)(a2 + 8) & 1;
  if ( v6 )
  {
    v7 = a2 + 16;
    v8 = 3;
  }
  else
  {
    v22 = *(unsigned int *)(a2 + 24);
    v7 = *(_QWORD *)(a2 + 16);
    if ( !(_DWORD)v22 )
      goto LABEL_20;
    v8 = v22 - 1;
  }
  v9 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (__int64 *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( v5 == *v10 )
    goto LABEL_4;
  v30 = 1;
  while ( v11 != -4096 )
  {
    v38 = v30 + 1;
    v9 = v8 & (v30 + v9);
    v10 = (__int64 *)(v7 + 24LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
      goto LABEL_4;
    v30 = v38;
  }
  if ( v6 )
  {
    v23 = 96;
    goto LABEL_21;
  }
  v22 = *(unsigned int *)(a2 + 24);
LABEL_20:
  v23 = 24 * v22;
LABEL_21:
  v10 = (__int64 *)(v7 + v23);
LABEL_4:
  v12 = 96;
  if ( !v6 )
    v12 = 24LL * *(unsigned int *)(a2 + 24);
  if ( v10 == (__int64 *)(v7 + v12) )
    return 0;
  v13 = *(_BYTE *)(a3 + 8) & 1;
  if ( v13 )
  {
    v15 = a3 + 16;
    v16 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(a3 + 24);
    v15 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v14 )
      goto LABEL_34;
    v16 = v14 - 1;
  }
  v17 = v16 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v18 = (__int64 *)(v15 + 24LL * v17);
  v19 = *v18;
  if ( a1 == *v18 )
    goto LABEL_11;
  v32 = 1;
  while ( v19 != -4096 )
  {
    v39 = v32 + 1;
    v17 = v16 & (v32 + v17);
    v18 = (__int64 *)(v15 + 24LL * v17);
    v19 = *v18;
    if ( a1 == *v18 )
      goto LABEL_11;
    v32 = v39;
  }
  if ( v13 )
  {
    v31 = 96;
    goto LABEL_35;
  }
  v14 = *(unsigned int *)(a3 + 24);
LABEL_34:
  v31 = 24 * v14;
LABEL_35:
  v18 = (__int64 *)(v15 + v31);
LABEL_11:
  v20 = 96;
  if ( !v13 )
    v20 = 24LL * *(unsigned int *)(a3 + 24);
  if ( v18 != (__int64 *)(v15 + v20) )
    return v18[1];
  v24 = *(_QWORD **)(a1 + 24);
  v25 = v10[1];
  v26 = *((_DWORD *)v10 + 4);
  v42 = &v24[*(unsigned int *)(a1 + 32)];
  v41 = v10[2];
  if ( v24 != v42 )
  {
    do
    {
      v43 = v24;
      v27 = sub_2965A20(*v24, a2, a3);
      if ( v28 == 1 )
        v26 = 1;
      v29 = __OFADD__(v27, v25);
      v25 += v27;
      if ( v29 )
      {
        v25 = 0x8000000000000000LL;
        if ( v27 > 0 )
          v25 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v24 = v43 + 1;
    }
    while ( v42 != v43 + 1 );
  }
  v46 = a1;
  v47.m128i_i64[0] = v25;
  v47.m128i_i64[1] = v26 | v41 & 0xFFFFFFFF00000000LL;
  if ( (unsigned __int8)sub_295D850(a3, &v46, &v44) )
    return v25;
  v33 = *(_DWORD *)(a3 + 8);
  v34 = v44;
  ++*(_QWORD *)a3;
  v45 = v34;
  v35 = (v33 >> 1) + 1;
  if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
  {
    v37 = 12;
    v36 = 4;
  }
  else
  {
    v36 = *(_DWORD *)(a3 + 24);
    v37 = 3 * v36;
  }
  if ( 4 * v35 >= v37 )
  {
    v40 = 2 * v36;
LABEL_53:
    sub_29655E0(a3, v40);
    sub_295D850(a3, &v46, &v45);
    v34 = v45;
    v35 = (*(_DWORD *)(a3 + 8) >> 1) + 1;
    goto LABEL_44;
  }
  if ( v36 - (v35 + *(_DWORD *)(a3 + 12)) <= v36 >> 3 )
  {
    v40 = v36;
    goto LABEL_53;
  }
LABEL_44:
  *(_DWORD *)(a3 + 8) = *(_DWORD *)(a3 + 8) & 1 | (2 * v35);
  if ( *(_QWORD *)v34 != -4096 )
    --*(_DWORD *)(a3 + 12);
  *(_QWORD *)v34 = v46;
  *(__m128i *)(v34 + 8) = _mm_loadu_si128(&v47);
  return v25;
}
