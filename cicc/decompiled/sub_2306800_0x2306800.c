// Function: sub_2306800
// Address: 0x2306800
//
_QWORD *__fastcall sub_2306800(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  unsigned __int64 v7; // r14
  __int64 v8; // r15
  unsigned int v9; // r13d
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __int16 v14; // ax
  __int16 v15; // ax
  __int64 v16; // rsi
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-100h]
  __int64 v22; // [rsp+8h] [rbp-F8h]
  __int64 v23; // [rsp+10h] [rbp-F0h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-E8h]
  __m128i v25; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v26; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-C0h]
  __int16 v28; // [rsp+42h] [rbp-BEh]
  __int64 v29; // [rsp+48h] [rbp-B8h]
  __int64 v30; // [rsp+50h] [rbp-B0h]
  __int64 v31; // [rsp+58h] [rbp-A8h]
  unsigned int v32; // [rsp+60h] [rbp-A0h]
  __m128i v33; // [rsp+80h] [rbp-80h] BYREF
  __m128i v34; // [rsp+90h] [rbp-70h] BYREF
  __int16 v35; // [rsp+A0h] [rbp-60h]
  __int16 v36; // [rsp+A2h] [rbp-5Eh]

  sub_D84AE0((__int64)&v23, a2 + 8, a3);
  v4 = v31;
  v5 = _mm_loadu_si128(&v25);
  v6 = _mm_loadu_si128(&v26);
  v31 = 0;
  v22 = v23;
  v7 = v24;
  v8 = v30;
  v24 = 0;
  v35 = v27;
  v9 = v32;
  v33 = v5;
  ++v29;
  v36 = v28;
  v21 = v4;
  v30 = 0;
  v32 = 0;
  v34 = v6;
  v10 = (_QWORD *)sub_22077B0(0x60u);
  v11 = v10;
  if ( v10 )
  {
    v12 = _mm_loadu_si128(&v33);
    v13 = _mm_loadu_si128(&v34);
    v10[2] = v7;
    v10[8] = 1;
    *v10 = &unk_4A0B2B8;
    v10[9] = v8;
    v10[1] = v22;
    v14 = v35;
    v11[10] = v21;
    *((_WORD *)v11 + 28) = v14;
    v15 = v36;
    *((_DWORD *)v11 + 22) = v9;
    *((_WORD *)v11 + 29) = v15;
    *(__m128i *)(v11 + 3) = v12;
    *(__m128i *)(v11 + 5) = v13;
    sub_C7D6A0(0, 0, 8);
  }
  else
  {
    sub_C7D6A0(v8, 16LL * v9, 8);
    if ( v7 )
    {
      v20 = *(_QWORD *)(v7 + 8);
      if ( v20 )
        j_j___libc_free_0(v20);
      j_j___libc_free_0(v7);
    }
  }
  v16 = v32;
  *a1 = v11;
  sub_C7D6A0(v30, 16 * v16, 8);
  v17 = v24;
  if ( v24 )
  {
    v18 = *(_QWORD *)(v24 + 8);
    if ( v18 )
      j_j___libc_free_0(v18);
    j_j___libc_free_0(v17);
  }
  return a1;
}
