// Function: sub_3843300
// Address: 0x3843300
//
__int64 *__fastcall sub_3843300(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r8
  int v4; // edx
  __int64 v7; // rax
  const __m128i *v8; // rbx
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rcx
  const __m128i *v11; // r13
  __m128i *v12; // rax
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rdx
  int *v16; // r13
  char v17; // r8
  __int64 v18; // rdi
  int v19; // esi
  unsigned int v20; // edx
  __int64 v21; // rax
  int v22; // r9d
  _BYTE *v23; // rbx
  __int64 v24; // rdx
  __int64 *v25; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // eax
  int v30; // r10d
  int v31; // [rsp+4h] [rbp-7Ch]
  int v32; // [rsp+8h] [rbp-78h]
  int v33; // [rsp+18h] [rbp-68h]
  int v34; // [rsp+2Ch] [rbp-54h] BYREF
  _BYTE *v35; // [rsp+30h] [rbp-50h] BYREF
  __int64 v36; // [rsp+38h] [rbp-48h]
  _BYTE v37[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a3;
  v4 = 0;
  v7 = *(unsigned int *)(a2 + 64);
  v8 = *(const __m128i **)(a2 + 40);
  v35 = v37;
  v7 *= 5;
  v9 = 0xCCCCCCCCCCCCCCCDLL * v7;
  v10 = 8 * v7;
  v36 = 0x100000000LL;
  v11 = (const __m128i *)((char *)v8 + 8 * v7);
  v12 = (__m128i *)v37;
  if ( v10 > 0x28 )
  {
    v31 = v3;
    v32 = v9;
    sub_C8D5F0((__int64)&v35, v37, v9, 0x10u, v3, v9);
    v4 = v36;
    LODWORD(v3) = v31;
    LODWORD(v9) = v32;
    v12 = (__m128i *)&v35[16 * (unsigned int)v36];
  }
  if ( v8 != v11 )
  {
    do
    {
      if ( v12 )
        *v12 = _mm_loadu_si128(v8);
      v8 = (const __m128i *)((char *)v8 + 40);
      ++v12;
    }
    while ( v11 != v8 );
    v4 = v36;
  }
  v13 = v9 + v4;
  v14 = (unsigned int)v3;
  v15 = *(_QWORD *)(a2 + 40);
  LODWORD(v36) = v13;
  v34 = sub_375D5B0(a1, *(_QWORD *)(v15 + 40LL * (unsigned int)v3), *(_QWORD *)(v15 + 40LL * (unsigned int)v3 + 8));
  v16 = sub_3805BC0(a1 + 712, &v34);
  sub_37593F0(a1, v16);
  v17 = *(_BYTE *)(a1 + 512) & 1;
  if ( v17 )
  {
    v18 = a1 + 520;
    v19 = 7;
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 528);
    v18 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v27 )
      goto LABEL_17;
    v19 = v27 - 1;
  }
  v20 = v19 & (37 * *v16);
  v21 = v18 + 24LL * v20;
  v22 = *(_DWORD *)v21;
  if ( *v16 == *(_DWORD *)v21 )
    goto LABEL_11;
  v29 = 1;
  while ( v22 != -1 )
  {
    v30 = v29 + 1;
    v20 = v19 & (v29 + v20);
    v21 = v18 + 24LL * v20;
    v22 = *(_DWORD *)v21;
    if ( *v16 == *(_DWORD *)v21 )
      goto LABEL_11;
    v29 = v30;
  }
  if ( v17 )
  {
    v28 = 192;
    goto LABEL_18;
  }
  v27 = *(unsigned int *)(a1 + 528);
LABEL_17:
  v28 = 24 * v27;
LABEL_18:
  v21 = v18 + v28;
LABEL_11:
  v23 = &v35[16 * v14];
  v33 = *(_DWORD *)(v21 + 16);
  *(_QWORD *)v23 = *(_QWORD *)(v21 + 8);
  v24 = (__int64)v35;
  *((_DWORD *)v23 + 2) = v33;
  v25 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, v24, (unsigned int)v36);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v25;
}
