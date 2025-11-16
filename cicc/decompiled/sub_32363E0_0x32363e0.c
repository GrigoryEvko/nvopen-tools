// Function: sub_32363E0
// Address: 0x32363e0
//
void __fastcall sub_32363E0(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        const __m128i *a4,
        int *a5,
        int *a6,
        unsigned __int8 *a7)
{
  __int64 v8; // rdi
  __int64 *v11; // r8
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r9
  _QWORD *v16; // rdx
  __int64 *v17; // r10
  __int64 v18; // r15
  int v19; // r14d
  int v20; // r13d
  __int16 v21; // bx
  __int64 v22; // rax
  __m128i v23; // xmm2
  _BYTE *v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 (__fastcall *v27)(_QWORD *, __int64); // rax
  __int64 v28; // rsi
  int v29; // eax
  __int64 *v30; // [rsp+0h] [rbp-70h]
  __int64 *v31; // [rsp+8h] [rbp-68h]
  __int64 *v32; // [rsp+8h] [rbp-68h]
  _QWORD *v33; // [rsp+10h] [rbp-60h]
  _QWORD *v34; // [rsp+10h] [rbp-60h]
  unsigned __int64 v35; // [rsp+10h] [rbp-60h]
  _QWORD *v36; // [rsp+10h] [rbp-60h]
  __m128i v38; // [rsp+20h] [rbp-50h] BYREF
  __m128i v39; // [rsp+30h] [rbp-40h] BYREF

  v8 = (__int64)(a1 + 12);
  v11 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a2 & 4) != 0 )
  {
    v25 = v11[4];
    v31 = a1;
    v35 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v39.m128i_i64[0] = v11[3];
    v39.m128i_i64[1] = v25;
    v26 = (_QWORD *)sub_3235E40(v8, &v39);
    v17 = v31;
    v16 = v26;
    if ( v26[3] != v26[2] )
      goto LABEL_3;
    *v26 = a2;
    v27 = (__int64 (__fastcall *)(_QWORD *, __int64))v31[18];
    v15 = *(_QWORD **)(v35 + 24);
    v28 = *(_QWORD *)(v35 + 32);
  }
  else
  {
    v13 = *v11;
    v30 = a1;
    v39.m128i_i64[0] = (__int64)(v11 + 4);
    v33 = v11 + 4;
    v39.m128i_i64[1] = v13;
    v14 = (_QWORD *)sub_3235E40(v8, &v39);
    v15 = v33;
    v16 = v14;
    v17 = v30;
    if ( v14[3] != v14[2] )
      goto LABEL_3;
    *v14 = a2;
    v27 = (__int64 (__fastcall *)(_QWORD *, __int64))v30[18];
    v28 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v32 = v17;
  v36 = v16;
  v29 = v27(v15, v28);
  v16 = v36;
  v17 = v32;
  *((_DWORD *)v36 + 2) = v29;
LABEL_3:
  v18 = *a3;
  v19 = *a5;
  v34 = v16;
  v20 = *a6;
  v38 = _mm_loadu_si128(a4);
  v21 = *a7;
  v22 = sub_A777F0(0x30u, v17);
  if ( v22 )
  {
    v39 = _mm_loadu_si128(&v38);
    *(_QWORD *)v22 = &unk_4A35768;
    *(_QWORD *)(v22 + 8) = v18;
    *(_BYTE *)(v22 + 16) = 1;
    v23 = _mm_loadu_si128(&v39);
    *(_WORD *)(v22 + 40) = v19;
    *(_WORD *)(v22 + 42) = v21 << 15;
    *(_DWORD *)(v22 + 44) = v20;
    *(__m128i *)(v22 + 24) = v23;
  }
  v39.m128i_i64[0] = v22;
  v24 = (_BYTE *)v34[3];
  if ( v24 == (_BYTE *)v34[4] )
  {
    sub_3226B30((__int64)(v34 + 2), v24, &v39);
  }
  else
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = v22;
      v24 = (_BYTE *)v34[3];
    }
    v34[3] = v24 + 8;
  }
}
