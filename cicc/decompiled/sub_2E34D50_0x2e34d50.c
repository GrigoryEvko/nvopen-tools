// Function: sub_2E34D50
// Address: 0x2e34d50
//
void __fastcall sub_2E34D50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v17; // dx
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r11
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 *v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 *v29; // rax
  __int64 *v30; // r13
  __int64 v31; // rbx
  __int64 *v32; // r12
  unsigned __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 *v35; // rdi
  int v36; // edx
  int v37; // r10d
  unsigned __int64 v38; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)a1;
  v9 = *a2;
  *(_QWORD *)(a1 + 80) += 32LL;
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 8) >= v11 + 32 && v8 )
  {
    *(_QWORD *)a1 = v11 + 32;
    v12 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v12 = sub_9D1E70(a1, 32, 32, 3);
    v11 = v12 & 0xFFFFFFFFFFFFFFF9LL;
  }
  *(_QWORD *)v12 = 0;
  *(_QWORD *)(v12 + 8) = 0;
  *(_QWORD *)(v12 + 16) = 0;
  *(_DWORD *)(v12 + 24) = 0;
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v10 + 24) + 8) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)(a2[6] & 0xFFFFFFFFFFFFFFF8LL) != a2 + 6 )
  {
    v14 = a2[7];
    for ( i = v14; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    for ( ; (*(_BYTE *)(v14 + 44) & 8) != 0; v14 = *(_QWORD *)(v14 + 8) )
      ;
    for ( j = *(_QWORD *)(v14 + 8); j != i; i = *(_QWORD *)(i + 8) )
    {
      v17 = *(_WORD *)(i + 68);
      if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
        break;
    }
    v18 = *(unsigned int *)(a1 + 144);
    v19 = *(_QWORD *)(a1 + 128);
    if ( (_DWORD)v18 )
    {
      a6 = (unsigned int)(v18 - 1);
      v20 = a6 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( i == *v21 )
      {
LABEL_15:
        v23 = (__int64 *)(v21[1] & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_16;
      }
      v36 = 1;
      while ( v22 != -4096 )
      {
        v37 = v36 + 1;
        v20 = a6 & (v36 + v20);
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( i == *v21 )
          goto LABEL_15;
        v36 = v37;
      }
    }
    v21 = (__int64 *)(v19 + 16 * v18);
    goto LABEL_15;
  }
  v23 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v10 + 24) + 8) & 0xFFFFFFFFFFFFFFF8LL);
LABEL_16:
  v24 = *v23;
  *(_QWORD *)(v12 + 8) = v23;
  v24 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v12 = v24;
  *(_QWORD *)(v24 + 8) = v12;
  *v23 = v12 | *v23 & 7;
  *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(int *)(v10 + 24) + 8) = v11;
  v25 = *(unsigned int *)(a1 + 160);
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
  {
    v38 = v13;
    sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v25 + 1, 0x10u, v13, a6);
    v25 = *(unsigned int *)(a1 + 160);
    v13 = v38;
  }
  v26 = (unsigned __int64 *)(*(_QWORD *)(a1 + 152) + 16 * v25);
  *v26 = v11;
  v26[1] = v13;
  v27 = *(unsigned int *)(a1 + 304);
  v28 = *(unsigned int *)(a1 + 308);
  ++*(_DWORD *)(a1 + 160);
  if ( v27 + 1 > v28 )
  {
    sub_C8D5F0(a1 + 296, (const void *)(a1 + 312), v27 + 1, 0x10u, v13, a6);
    v27 = *(unsigned int *)(a1 + 304);
  }
  v29 = (unsigned __int64 *)(*(_QWORD *)(a1 + 296) + 16 * v27);
  *v29 = v11;
  v29[1] = (unsigned __int64)a2;
  ++*(_DWORD *)(a1 + 304);
  sub_2FAD5B0(a1);
  v30 = *(__int64 **)(a1 + 296);
  v31 = 2LL * *(unsigned int *)(a1 + 304);
  v32 = &v30[v31];
  if ( v30 != &v30[v31] )
  {
    _BitScanReverse64(&v33, (v31 * 8) >> 4);
    sub_2E34B10(v30, (char *)&v30[v31], 2LL * (int)(63 - (v33 ^ 0x3F)));
    if ( (unsigned __int64)v31 <= 32 )
    {
      sub_2E30100(v30, &v30[v31]);
    }
    else
    {
      v34 = v30 + 32;
      sub_2E30100(v30, v30 + 32);
      if ( v32 != v30 + 32 )
      {
        do
        {
          v35 = v34;
          v34 += 2;
          sub_2E30090(v35);
        }
        while ( v32 != v34 );
      }
    }
  }
}
