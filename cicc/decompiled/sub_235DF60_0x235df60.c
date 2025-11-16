// Function: sub_235DF60
// Address: 0x235df60
//
__int64 __fastcall sub_235DF60(unsigned __int64 *a1, __m128i *a2)
{
  __int64 (__fastcall *v2)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v3; // rdx
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v7; // xmm2
  __int64 v8; // rax
  __int64 v9; // rcx
  __m128i v10; // xmm0
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __int64 (__fastcall *v13)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // [rsp+8h] [rbp-58h] BYREF
  __m128i v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v18)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-40h]
  __int64 v19; // [rsp+28h] [rbp-38h]
  __m128i v20; // [rsp+30h] [rbp-30h] BYREF
  char v21; // [rsp+40h] [rbp-20h]

  v2 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a2[1].m128i_i64[0];
  v3 = v19;
  a2[1].m128i_i64[0] = 0;
  v4 = _mm_loadu_si128(&v17);
  v5 = _mm_loadu_si128(a2);
  v18 = v2;
  v6 = a2[1].m128i_i64[1];
  a2[1].m128i_i64[1] = v3;
  v7 = _mm_loadu_si128(a2 + 2);
  v19 = v6;
  LOBYTE(v6) = a2[3].m128i_i8[0];
  *a2 = v4;
  v21 = v6;
  v17 = v5;
  v20 = v7;
  v8 = sub_22077B0(0x40u);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v8 + 32);
    v10 = _mm_loadu_si128(&v17);
    v11 = _mm_loadu_si128((const __m128i *)(v8 + 8));
    v12 = _mm_loadu_si128(&v20);
    *(_QWORD *)v8 = &unk_4A15678;
    v13 = v18;
    v18 = 0;
    *(_QWORD *)(v8 + 24) = v13;
    v14 = v19;
    v19 = v9;
    *(_QWORD *)(v8 + 32) = v14;
    v17 = v11;
    *(_BYTE *)(v8 + 56) = v21;
    *(__m128i *)(v8 + 8) = v10;
    *(__m128i *)(v8 + 40) = v12;
  }
  v16 = v8;
  sub_235DE40(a1, (unsigned __int64 *)&v16);
  if ( v16 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
  result = (__int64)v18;
  if ( v18 )
    return v18(&v17, &v17, 3);
  return result;
}
