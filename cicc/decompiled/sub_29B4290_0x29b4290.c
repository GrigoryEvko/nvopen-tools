// Function: sub_29B4290
// Address: 0x29b4290
//
__int64 __fastcall sub_29B4290(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v5; // r14
  __m128i *v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  unsigned __int64 v10; // xmm0_8
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rsi
  const void *v17; // [rsp+8h] [rbp-158h]
  __int64 v18; // [rsp+10h] [rbp-150h]
  __int64 i; // [rsp+20h] [rbp-140h]
  unsigned __int64 v20; // [rsp+30h] [rbp-130h]
  unsigned __int64 v21; // [rsp+40h] [rbp-120h]
  _BYTE v22[16]; // [rsp+50h] [rbp-110h] BYREF
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64, __int64, __int64); // [rsp+60h] [rbp-100h]
  unsigned __int8 (__fastcall *v24)(_BYTE *, __int64, __int64, __int64, __int64); // [rsp+68h] [rbp-F8h]
  __m128i v25; // [rsp+70h] [rbp-F0h]
  __m128i v26; // [rsp+80h] [rbp-E0h]
  _BYTE v27[16]; // [rsp+90h] [rbp-D0h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64, __int64, __int64); // [rsp+A0h] [rbp-C0h]
  __int64 v29; // [rsp+A8h] [rbp-B8h]
  __m128i v30; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+C0h] [rbp-A0h] BYREF
  _BYTE v32[16]; // [rsp+D0h] [rbp-90h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+E0h] [rbp-80h]
  unsigned __int8 (__fastcall *v34)(_BYTE *, __int64, __int64, __int64, __int64); // [rsp+E8h] [rbp-78h]
  __m128i v35; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v36; // [rsp+100h] [rbp-60h] BYREF
  _BYTE v37[16]; // [rsp+110h] [rbp-50h] BYREF
  void (__fastcall *v38)(_BYTE *, _BYTE *, __int64); // [rsp+120h] [rbp-40h]
  __int64 v39; // [rsp+128h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  v17 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  result = a2 + 72;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  v3 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  for ( i = a2 + 72; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    v5 = v3 - 24;
    v6 = &v30;
    if ( !v3 )
      v5 = 0;
    sub_AA72C0(&v30, v5, 1);
    v10 = _mm_loadu_si128(&v30).m128i_u64[0];
    v23 = 0;
    v20 = v10;
    v21 = _mm_loadu_si128(&v31).m128i_u64[0];
    if ( v33 )
    {
      v6 = (__m128i *)v22;
      v33(v22, v32, 2);
      v24 = v34;
      v23 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64))v33;
    }
    v11 = _mm_loadu_si128(&v35);
    v12 = _mm_loadu_si128(&v36);
    v28 = 0;
    v25 = v11;
    v26 = v12;
    if ( v38 )
    {
      v6 = (__m128i *)v27;
      v38(v27, v37, 2);
      v29 = v39;
      v28 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64))v38;
    }
    while ( 1 )
    {
      v13 = v20;
      v14 = v20;
      if ( v20 == v25.m128i_i64[0] )
        break;
      while ( 1 )
      {
        if ( !v14 )
          BUG();
        if ( *(_BYTE *)(v14 - 24) == 60 )
        {
          v15 = *(unsigned int *)(a1 + 8);
          if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            v6 = (__m128i *)a1;
            v18 = v14;
            sub_C8D5F0(a1, v17, v15 + 1, 8u, v14, v9);
            v15 = *(unsigned int *)(a1 + 8);
            v14 = v18;
          }
          v7 = *(_QWORD *)a1;
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v15) = v14 - 24;
          v13 = v20;
          ++*(_DWORD *)(a1 + 8);
        }
        v13 = *(_QWORD *)(v13 + 8);
        v8 = 0;
        v20 = v13;
        v14 = v13;
        v16 = v13;
        if ( v13 != v21 )
          break;
LABEL_21:
        if ( v14 == v25.m128i_i64[0] )
          goto LABEL_22;
      }
      while ( 1 )
      {
        if ( v16 )
          v16 -= 24;
        if ( !v23 )
          sub_4263D6(v6, v16, v7);
        v6 = (__m128i *)v22;
        if ( v24(v22, v16, v7, v8, v14) )
          break;
        v7 = 0;
        v16 = *(_QWORD *)(v20 + 8);
        v20 = v16;
        v13 = v16;
        if ( v21 == v16 )
        {
          v14 = v16;
          goto LABEL_21;
        }
      }
    }
LABEL_22:
    if ( v28 )
      v28(v27, v27, 3, v8, v14);
    if ( v23 )
      v23(v22, v22, 3, v8, v14);
    if ( v38 )
      ((void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64))v38)(v37, v37, 3, v8, v14);
    if ( v33 )
      ((void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64))v33)(v32, v32, 3, v8, v14);
    result = sub_29B38D0(a1, v5);
  }
  return result;
}
