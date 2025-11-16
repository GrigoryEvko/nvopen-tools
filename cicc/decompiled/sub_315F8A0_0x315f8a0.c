// Function: sub_315F8A0
// Address: 0x315f8a0
//
__int64 __fastcall sub_315F8A0(__int64 a1, __int64 a2, __int16 a3, char a4, char a5)
{
  __int64 v7; // r15
  char v8; // al
  __int64 v9; // rdi
  const __m128i *v10; // r15
  char v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  unsigned __int8 v16; // al
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rdx
  __m128i *v24; // rax
  __int64 v25; // r12
  unsigned __int8 v26; // r9
  char v27; // cl
  __int64 v28; // rdi
  const void *v29; // rsi
  char *v30; // r15
  unsigned __int64 v33; // [rsp+10h] [rbp-60h] BYREF
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  char v37; // [rsp+30h] [rbp-40h]
  unsigned __int8 v38; // [rsp+31h] [rbp-3Fh]
  __int64 v39; // [rsp+38h] [rbp-38h]

  v7 = *(_QWORD *)a1;
  v8 = sub_AE5020(*(_QWORD *)a1, a2);
  v9 = v7;
  v10 = (const __m128i *)&v33;
  v11 = v8;
  v12 = sub_9208B0(v9, a2);
  v34 = v13;
  v33 = ((1LL << v11) + ((unsigned __int64)(v12 + 7) >> 3) - 1) >> v11 << v11;
  v14 = sub_CA1930(&v33);
  if ( !v14 )
    return v14;
  v16 = sub_AE5020(*(_QWORD *)a1, a2);
  if ( !a5 )
  {
    if ( !HIBYTE(a3) )
      LOBYTE(a3) = v16;
    v18 = 0;
    if ( !*(_BYTE *)(a1 + 27) )
      goto LABEL_8;
    v26 = *(_BYTE *)(a1 + 26);
    goto LABEL_16;
  }
  if ( *(_BYTE *)(a1 + 27) )
  {
    v26 = *(_BYTE *)(a1 + 26);
    if ( v16 > v26 )
      v16 = *(_BYTE *)(a1 + 26);
    if ( !HIBYTE(a3) )
      LOBYTE(a3) = v16;
LABEL_16:
    v18 = 0;
    if ( (unsigned __int8)a3 > v26 )
    {
      v27 = a3;
      LOBYTE(a3) = v26;
      v18 = (((1LL << v27) + (1LL << v26) - 1) & -(1LL << v27)) - (1LL << v26);
      v14 += v18;
    }
    goto LABEL_8;
  }
  if ( !HIBYTE(a3) )
    LOBYTE(a3) = v16;
  v18 = 0;
LABEL_8:
  v19 = -1;
  if ( a4 )
  {
    v19 = -(1LL << a3) & ((1LL << a3) + *(_QWORD *)(a1 + 16) - 1);
    *(_QWORD *)(a1 + 16) = v14 + v19;
  }
  v38 = v16;
  v20 = *(unsigned int *)(a1 + 40);
  v34 = v19;
  v21 = *(unsigned int *)(a1 + 44);
  v22 = v20 + 1;
  v33 = v14;
  v23 = *(_QWORD *)(a1 + 32);
  v35 = a2;
  v36 = 0;
  v37 = a3;
  v39 = v18;
  if ( v20 + 1 > v21 )
  {
    v28 = a1 + 32;
    v29 = (const void *)(a1 + 48);
    if ( v23 > (unsigned __int64)&v33 || (unsigned __int64)&v33 >= v23 + 48 * v20 )
    {
      sub_C8D5F0(v28, v29, v22, 0x30u, v17, v22);
      v23 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
    }
    else
    {
      v30 = (char *)&v33 - v23;
      sub_C8D5F0(v28, v29, v22, 0x30u, v17, v22);
      v23 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v10 = (const __m128i *)&v30[v23];
    }
  }
  v24 = (__m128i *)(v23 + 48 * v20);
  *v24 = _mm_loadu_si128(v10);
  v24[1] = _mm_loadu_si128(v10 + 1);
  v24[2] = _mm_loadu_si128(v10 + 2);
  v25 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = v25;
  return v25 - 1;
}
