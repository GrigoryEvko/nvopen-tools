// Function: sub_235AA40
// Address: 0x235aa40
//
__int64 __fastcall sub_235AA40(unsigned __int64 *a1, __m128i *a2)
{
  __int64 (__fastcall *v2)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v3; // rdx
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __m128i v9; // xmm0
  __m128i v10; // xmm2
  __int64 (__fastcall *v11)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __m128i v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v16)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-30h]
  __int64 v17; // [rsp+28h] [rbp-28h]
  char v18; // [rsp+30h] [rbp-20h]

  v2 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a2[1].m128i_i64[0];
  v3 = v17;
  a2[1].m128i_i64[0] = 0;
  v4 = _mm_loadu_si128(&v15);
  v5 = _mm_loadu_si128(a2);
  v16 = v2;
  v6 = a2[1].m128i_i64[1];
  a2[1].m128i_i64[1] = v3;
  v17 = v6;
  LOBYTE(v6) = a2[2].m128i_i8[0];
  *a2 = v4;
  v18 = v6;
  v15 = v5;
  v7 = sub_22077B0(0x30u);
  if ( v7 )
  {
    v8 = *(_QWORD *)(v7 + 32);
    v9 = _mm_loadu_si128(&v15);
    v10 = _mm_loadu_si128((const __m128i *)(v7 + 8));
    *(_QWORD *)v7 = &unk_4A0EC38;
    v11 = v16;
    v16 = 0;
    *(_QWORD *)(v7 + 24) = v11;
    v12 = v17;
    v17 = v8;
    *(_QWORD *)(v7 + 32) = v12;
    v15 = v10;
    *(_BYTE *)(v7 + 40) = v18;
    *(__m128i *)(v7 + 8) = v9;
  }
  v14 = v7;
  sub_235A870(a1, (unsigned __int64 *)&v14);
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  result = (__int64)v16;
  if ( v16 )
    return v16(&v15, &v15, 3);
  return result;
}
