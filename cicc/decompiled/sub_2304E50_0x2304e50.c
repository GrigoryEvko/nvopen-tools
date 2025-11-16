// Function: sub_2304E50
// Address: 0x2304e50
//
__int64 *__fastcall sub_2304E50(__int64 *a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  int v17; // eax
  int v18; // eax
  char *v19; // rdi
  __m128i v21; // [rsp+0h] [rbp-120h] BYREF
  __m128i v22; // [rsp+10h] [rbp-110h] BYREF
  int v23; // [rsp+20h] [rbp-100h]
  __m128i v24; // [rsp+24h] [rbp-FCh] BYREF
  char *v25; // [rsp+38h] [rbp-E8h] BYREF
  unsigned int v26; // [rsp+40h] [rbp-E0h]
  char v27; // [rsp+48h] [rbp-D8h] BYREF
  __m128i v28; // [rsp+80h] [rbp-A0h] BYREF
  __m128i v29; // [rsp+90h] [rbp-90h] BYREF
  int v30; // [rsp+A0h] [rbp-80h]
  __m128i v31; // [rsp+A4h] [rbp-7Ch] BYREF
  char *v32; // [rsp+B8h] [rbp-68h] BYREF
  __int64 v33; // [rsp+C0h] [rbp-60h]
  _BYTE v34[88]; // [rsp+C8h] [rbp-58h] BYREF

  sub_30C34B0(&v21, a2 + 8);
  v6 = _mm_loadu_si128(&v21);
  v32 = v34;
  v7 = _mm_loadu_si128(&v22);
  v8 = _mm_loadu_si128(&v24);
  v30 = v23;
  v33 = 0x200000000LL;
  v28 = v6;
  v29 = v7;
  v31 = v8;
  if ( v26 )
    sub_2303280((__int64)&v32, &v25, v26, v3, v4, v5);
  v9 = sub_22077B0(0x80u);
  v14 = v9;
  if ( v9 )
  {
    v15 = _mm_loadu_si128(&v29);
    v16 = _mm_loadu_si128(&v31);
    *(__m128i *)(v9 + 8) = _mm_loadu_si128(&v28);
    *(_QWORD *)v9 = &unk_4A0B448;
    v17 = v30;
    *(__m128i *)(v14 + 24) = v15;
    *(_DWORD *)(v14 + 40) = v17;
    *(_QWORD *)(v14 + 64) = v14 + 80;
    *(_QWORD *)(v14 + 72) = 0x200000000LL;
    v18 = v33;
    *(__m128i *)(v14 + 44) = v16;
    if ( v18 )
      sub_2303280(v14 + 64, &v32, v10, v11, v12, v13);
  }
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  v19 = v25;
  *a1 = v14;
  if ( v19 != &v27 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
