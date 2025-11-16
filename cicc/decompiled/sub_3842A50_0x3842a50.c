// Function: sub_3842A50
// Address: 0x3842a50
//
__int64 *__fastcall sub_3842A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r8
  int v7; // edx
  __int64 v9; // rax
  const __m128i *v10; // rbx
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rcx
  const __m128i *v13; // r12
  __m128i *v14; // rax
  int *v15; // rbx
  char v16; // si
  __int64 v17; // r9
  int v18; // ecx
  unsigned int v19; // edi
  __int64 v20; // rax
  int v21; // r10d
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 *v24; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // eax
  int v29; // r11d
  __int64 *v30; // [rsp+8h] [rbp-D8h]
  int v31; // [rsp+18h] [rbp-C8h]
  int v32; // [rsp+2Ch] [rbp-B4h] BYREF
  _BYTE *v33; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-A8h]
  _BYTE v35[160]; // [rsp+40h] [rbp-A0h] BYREF

  v6 = (__int64 *)a2;
  v7 = 0;
  v9 = *(unsigned int *)(a2 + 64);
  v10 = *(const __m128i **)(a2 + 40);
  v33 = v35;
  v9 *= 5;
  v11 = 0xCCCCCCCCCCCCCCCDLL * v9;
  v12 = 8 * v9;
  v34 = 0x700000000LL;
  v13 = (const __m128i *)((char *)v10 + 8 * v9);
  v14 = (__m128i *)v35;
  if ( v12 > 0x118 )
  {
    sub_C8D5F0((__int64)&v33, v35, v11, 0x10u, a2, a6);
    v7 = v34;
    v6 = (__int64 *)a2;
    v14 = (__m128i *)&v33[16 * (unsigned int)v34];
  }
  if ( v10 != v13 )
  {
    do
    {
      if ( v14 )
        *v14 = _mm_loadu_si128(v10);
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v14;
    }
    while ( v13 != v10 );
    v7 = v34;
  }
  v30 = v6;
  LODWORD(v34) = v11 + v7;
  v32 = sub_375D5B0(a1, *(_QWORD *)(v6[5] + 40), *(_QWORD *)(v6[5] + 48));
  v15 = sub_3805BC0(a1 + 712, &v32);
  sub_37593F0(a1, v15);
  v16 = *(_BYTE *)(a1 + 512) & 1;
  if ( v16 )
  {
    v17 = a1 + 520;
    v18 = 7;
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 528);
    v17 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v26 )
      goto LABEL_17;
    v18 = v26 - 1;
  }
  v19 = v18 & (37 * *v15);
  v20 = v17 + 24LL * v19;
  v21 = *(_DWORD *)v20;
  if ( *v15 == *(_DWORD *)v20 )
    goto LABEL_11;
  v28 = 1;
  while ( v21 != -1 )
  {
    v29 = v28 + 1;
    v19 = v18 & (v28 + v19);
    v20 = v17 + 24LL * v19;
    v21 = *(_DWORD *)v20;
    if ( *v15 == *(_DWORD *)v20 )
      goto LABEL_11;
    v28 = v29;
  }
  if ( v16 )
  {
    v27 = 192;
    goto LABEL_18;
  }
  v26 = *(unsigned int *)(a1 + 528);
LABEL_17:
  v27 = 24 * v26;
LABEL_18:
  v20 = v17 + v27;
LABEL_11:
  v22 = (__int64)v33;
  v31 = *(_DWORD *)(v20 + 16);
  *((_QWORD *)v33 + 2) = *(_QWORD *)(v20 + 8);
  v23 = (unsigned int)v34;
  *(_DWORD *)(v22 + 24) = v31;
  v24 = sub_33EC210(*(_QWORD **)(a1 + 8), v30, v22, v23);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return v24;
}
