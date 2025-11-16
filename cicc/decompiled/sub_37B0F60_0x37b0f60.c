// Function: sub_37B0F60
// Address: 0x37b0f60
//
__int64 __fastcall sub_37B0F60(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // r9
  __int64 v7; // rax
  unsigned int v9; // esi
  int v10; // eax
  __int64 v11; // r13
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rcx
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __m128i *v20; // rsi
  __m128i *v21; // rdi
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  _BYTE *v24; // rdi
  unsigned __int64 v25; // r14
  __int64 v26; // rdi
  __m128i v27; // [rsp+0h] [rbp-100h] BYREF
  __m128i v28; // [rsp+10h] [rbp-F0h]
  int v29; // [rsp+20h] [rbp-E0h]
  _QWORD v30[2]; // [rsp+30h] [rbp-D0h] BYREF
  char v31; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v32; // [rsp+70h] [rbp-90h] BYREF
  __m128i v33; // [rsp+80h] [rbp-80h]
  _BYTE *v34; // [rsp+90h] [rbp-70h]
  __int64 v35; // [rsp+98h] [rbp-68h]
  _BYTE v36[96]; // [rsp+A0h] [rbp-60h] BYREF

  v4 = _mm_loadu_si128(a2);
  v5 = _mm_loadu_si128(a2 + 1);
  v29 = 0;
  v32 = v4;
  v33 = v5;
  v27 = v4;
  v28 = v5;
  if ( (unsigned __int8)sub_3794400(a1, v27.m128i_i64, v30) )
  {
    v7 = *(unsigned int *)(v30[0] + 32LL);
    return *(_QWORD *)(a1 + 32) + 96 * v7 + 32;
  }
  v9 = *(_DWORD *)(a1 + 24);
  v10 = *(_DWORD *)(a1 + 16);
  v11 = v30[0];
  ++*(_QWORD *)a1;
  v12 = v10 + 1;
  v13 = 2 * v9;
  v32.m128i_i64[0] = v11;
  if ( 4 * v12 >= 3 * v9 )
  {
    sub_37B0D60(a1, v13);
  }
  else
  {
    if ( v9 - *(_DWORD *)(a1 + 20) - v12 > v9 >> 3 )
      goto LABEL_6;
    sub_37B0D60(a1, v9);
  }
  sub_3794400(a1, v27.m128i_i64, &v32);
  v11 = v32.m128i_i64[0];
  v12 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *(_QWORD *)v11 || *(_DWORD *)(v11 + 8) != -1 || *(_QWORD *)(v11 + 16) || *(_DWORD *)(v11 + 24) != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v11 = v27.m128i_i64[0];
  *(_DWORD *)(v11 + 8) = v27.m128i_i32[2];
  *(_QWORD *)(v11 + 16) = v28.m128i_i64[0];
  *(_DWORD *)(v11 + 24) = v28.m128i_i32[2];
  *(_DWORD *)(v11 + 32) = v29;
  v14 = *(unsigned int *)(a1 + 40);
  v15 = _mm_loadu_si128(a2);
  v16 = _mm_loadu_si128(a2 + 1);
  v30[0] = &v31;
  v17 = *(unsigned int *)(a1 + 44);
  v18 = v14 + 1;
  v34 = v36;
  v30[1] = 0xC00000000LL;
  v35 = 0xC00000000LL;
  v7 = v14;
  v32 = v15;
  v33 = v16;
  if ( v14 + 1 > v17 )
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = a1 + 32;
    if ( v25 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v25 + 96 * v14 )
    {
      sub_37945D0(v26, v18, v17, v14, v13, v6);
      v14 = *(unsigned int *)(a1 + 40);
      v19 = *(_QWORD *)(a1 + 32);
      v20 = &v32;
      v7 = v14;
    }
    else
    {
      sub_37945D0(v26, v18, v17, v14, v13, v6);
      v19 = *(_QWORD *)(a1 + 32);
      v14 = *(unsigned int *)(a1 + 40);
      v20 = (__m128i *)((char *)&v32 + v19 - v25);
      v7 = v14;
    }
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = &v32;
  }
  v21 = (__m128i *)(v19 + 96 * v14);
  if ( v21 )
  {
    v22 = _mm_loadu_si128(v20);
    v23 = _mm_loadu_si128(v20 + 1);
    v21[2].m128i_i64[0] = (__int64)v21[3].m128i_i64;
    v21[2].m128i_i64[1] = 0xC00000000LL;
    *v21 = v22;
    v21[1] = v23;
    if ( v20[2].m128i_i32[2] )
      sub_37748E0((__int64)v21[2].m128i_i64, (char **)&v20[2], v19, v14, v13, v6);
    v7 = *(unsigned int *)(a1 + 40);
  }
  v24 = v34;
  *(_DWORD *)(a1 + 40) = v7 + 1;
  if ( v24 != v36 )
  {
    _libc_free((unsigned __int64)v24);
    v7 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v11 + 32) = v7;
  return *(_QWORD *)(a1 + 32) + 96 * v7 + 32;
}
