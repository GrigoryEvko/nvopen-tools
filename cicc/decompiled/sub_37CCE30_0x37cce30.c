// Function: sub_37CCE30
// Address: 0x37cce30
//
__int64 __fastcall sub_37CCE30(__int64 a1, const __m128i *a2, __int64 a3)
{
  __m128i *v4; // r14
  __m128i v7; // xmm1
  __int64 v8; // rax
  unsigned int v9; // r13d
  unsigned int v11; // esi
  int v12; // eax
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  __m128i *v24; // rax
  unsigned __int64 v25; // r12
  __int64 v26; // rdi
  const void *v27; // rsi
  __m128i *v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v30; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v31[3]; // [rsp+20h] [rbp-90h] BYREF
  char v32; // [rsp+38h] [rbp-78h]
  __int64 v33; // [rsp+40h] [rbp-70h]
  __m128i v34; // [rsp+50h] [rbp-60h] BYREF
  __m128i v35; // [rsp+60h] [rbp-50h] BYREF
  __int64 v36; // [rsp+70h] [rbp-40h]
  __int64 v37; // [rsp+78h] [rbp-38h]

  v4 = &v34;
  v7 = _mm_loadu_si128(a2 + 1);
  v8 = a2[2].m128i_i64[0];
  v9 = *(_DWORD *)(a1 + 16);
  v34 = _mm_loadu_si128(a2);
  v36 = v8;
  LODWORD(v37) = v9;
  v35 = v7;
  if ( (unsigned __int8)sub_37BE800(a1, (__int64)&v34, &v29) )
    return *(unsigned int *)(v29 + 40);
  v11 = *(_DWORD *)(a1 + 24);
  v12 = *(_DWORD *)(a1 + 16);
  v13 = v29;
  ++*(_QWORD *)a1;
  v14 = v12 + 1;
  v30 = v13;
  if ( 4 * v14 >= 3 * v11 )
  {
    v11 *= 2;
    goto LABEL_12;
  }
  if ( v11 - *(_DWORD *)(a1 + 20) - v14 <= v11 >> 3 )
  {
LABEL_12:
    sub_37CCC90(a1, v11);
    sub_37BE800(a1, (__int64)&v34, &v30);
    v13 = v30;
    v14 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v14;
  v28 = (__m128i *)v13;
  v31[0] = 0;
  v32 = 0;
  v33 = 0;
  if ( !sub_F34140(v13, (__int64)v31) )
    --*(_DWORD *)(a1 + 20);
  *v28 = _mm_loadu_si128(&v34);
  v28[1] = _mm_loadu_si128(&v35);
  v28[2].m128i_i64[0] = v36;
  v28[2].m128i_i32[2] = v37;
  v17 = a2[2].m128i_i64[0];
  v18 = _mm_loadu_si128(a2);
  v19 = _mm_loadu_si128(a2 + 1);
  v37 = a3;
  v36 = v17;
  v20 = *(unsigned int *)(a1 + 40);
  v21 = *(unsigned int *)(a1 + 44);
  v34 = v18;
  v22 = v20 + 1;
  v35 = v19;
  if ( v20 + 1 > v21 )
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = a1 + 32;
    v27 = (const void *)(a1 + 48);
    if ( v25 <= (unsigned __int64)&v34 && (unsigned __int64)&v34 < v25 + 48 * v20 )
    {
      sub_C8D5F0(v26, v27, v22, 0x30u, v15, v16);
      v23 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v4 = (__m128i *)((char *)&v34 + v23 - v25);
    }
    else
    {
      sub_C8D5F0(v26, v27, v22, 0x30u, v15, v16);
      v23 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 32);
  }
  v24 = (__m128i *)(v23 + 48 * v20);
  *v24 = _mm_loadu_si128(v4);
  v24[1] = _mm_loadu_si128(v4 + 1);
  v24[2] = _mm_loadu_si128(v4 + 2);
  ++*(_DWORD *)(a1 + 40);
  return v9;
}
