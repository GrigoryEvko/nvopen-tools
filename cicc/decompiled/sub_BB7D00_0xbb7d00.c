// Function: sub_BB7D00
// Address: 0xbb7d00
//
__int64 __fastcall sub_BB7D00(__int64 a1, int *a2, _BYTE *a3, __int64 a4, _QWORD *a5)
{
  int v6; // eax
  void (__fastcall *v7)(__m128i *, __int64, __int64); // rax
  __int64 v9; // rax
  __m128i *v10; // rax
  void (__fastcall *v11)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v12; // rcx
  __m128i v13; // xmm0
  __m128i v14; // xmm2
  __int64 v15; // rdx
  __int64 (__fastcall *v16)(__int64 *, unsigned int *); // rdx
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rcx
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  void (__fastcall *v20)(_QWORD, _QWORD, _QWORD); // rax
  __int64 (__fastcall *v21)(__int64 *, unsigned int *); // rcx
  __int64 result; // rax
  __m128i v23; // [rsp+0h] [rbp-80h] BYREF
  void (__fastcall *v24)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  _QWORD v26[2]; // [rsp+20h] [rbp-60h] BYREF
  void (__fastcall *v27)(_QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-50h]
  __int64 (__fastcall *v28)(__int64 *, unsigned int *); // [rsp+38h] [rbp-48h]
  __m128i v29; // [rsp+40h] [rbp-40h] BYREF
  void (__fastcall *v30)(_QWORD, _QWORD, _QWORD); // [rsp+50h] [rbp-30h]
  __int64 (__fastcall *v31)(__int64 *, unsigned int *); // [rsp+58h] [rbp-28h]

  v6 = *a2;
  *(_BYTE *)(a1 + 156) = 1;
  v24 = 0;
  *(_DWORD *)(a1 + 136) = v6;
  *(_DWORD *)(a1 + 152) = *a2;
  *(_BYTE *)(a1 + 12) = *a3 & 7 | *(_BYTE *)(a1 + 12) & 0xF8;
  v7 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a4 + 16);
  if ( !v7 )
  {
    v27 = 0;
    goto LABEL_14;
  }
  v7(&v23, a4, 2);
  v9 = *(_QWORD *)(a4 + 24);
  v27 = 0;
  v25 = v9;
  v24 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a4 + 16);
  if ( !v24 )
  {
LABEL_14:
    v16 = v31;
    v17 = 0;
    goto LABEL_6;
  }
  v10 = (__m128i *)sub_22077B0(32);
  if ( v10 )
  {
    v11 = v24;
    v12 = v10[1].m128i_i64[1];
    v24 = 0;
    v13 = _mm_loadu_si128(&v23);
    v14 = _mm_loadu_si128(v10);
    v10[1].m128i_i64[0] = (__int64)v11;
    v15 = v25;
    v25 = v12;
    v10[1].m128i_i64[1] = v15;
    v23 = v14;
    *v10 = v13;
  }
  v26[0] = v10;
  v30 = 0;
  v28 = sub_BB74E0;
  v27 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))sub_BB7EC0;
  sub_BB7EC0(&v29, v26, 2);
  v16 = v28;
  v17 = v27;
LABEL_6:
  v18 = _mm_loadu_si128(&v29);
  v19 = _mm_loadu_si128((const __m128i *)(a1 + 168));
  v20 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a1 + 184);
  *(_QWORD *)(a1 + 184) = v17;
  v21 = *(__int64 (__fastcall **)(__int64 *, unsigned int *))(a1 + 192);
  v29 = v19;
  v30 = v20;
  v31 = v21;
  *(_QWORD *)(a1 + 192) = v16;
  *(__m128i *)(a1 + 168) = v18;
  if ( v20 )
    v20(&v29, &v29, 3);
  if ( v27 )
    v27(v26, v26, 3);
  if ( v24 )
    v24(&v23, &v23, 3);
  result = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = result;
  return result;
}
