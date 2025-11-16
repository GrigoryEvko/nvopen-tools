// Function: sub_65F0A0
// Address: 0x65f0a0
//
__int64 __fastcall sub_65F0A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rcx
  __int64 *v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int8 v15; // cl
  char v16; // r14
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-D8h]
  __m128i v23; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v24; // [rsp+20h] [rbp-C0h]
  __m128i v25; // [rsp+30h] [rbp-B0h]
  __m128i v26; // [rsp+40h] [rbp-A0h]
  _QWORD v27[11]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v28; // [rsp+A8h] [rbp-38h] BYREF

  v3 = (__int64 *)a2;
  if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
  {
    a2 = 2888;
    sub_684B40(&dword_4F063F8, 2888);
  }
  memset(v27, 0, sizeof(v27));
  v5 = &v28;
  v4 = 0;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  v6 = *(_QWORD *)(a1 + 24);
  v27[0] = *(_QWORD *)&dword_4F063F8;
  v27[4] = v6;
  v7 = *v3;
  v27[2] = *(_QWORD *)&dword_4F063F8;
  v27[6] = *(_QWORD *)&dword_4F063F8;
  v27[5] = v7;
  v8 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v9 = qword_4F063F0;
  v10 = _mm_loadu_si128(&xmmword_4D04A20);
  v11 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  v24 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
  v27[3] = qword_4F063F0;
  v27[7] = qword_4F063F0;
  v23 = v8;
  v25 = v10;
  v26 = v11;
  if ( (v24.m128i_i8[0] & 0x49) != 0 )
  {
    a2 = (__int64)&dword_4F063F8;
    v5 = (__int64 *)((v24.m128i_i8[0] & 1) == 0 ? 502 : 283);
    sub_6851C0(v5, &dword_4F063F8);
    v24 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v23.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v24.m128i_i8[1] |= 0x20u;
    v25 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v23.m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v26 = _mm_loadu_si128(&xmmword_4F06660[3]);
  }
  sub_7B8B50(v5, a2, v9, v4);
  sub_6729D0(a1);
  v12 = sub_5CC190(6);
  if ( v12 )
    v27[7] = unk_4F061D8;
  *(_QWORD *)(a1 + 124) = *(_QWORD *)(a1 + 124) & 0xDFFFFFFFFFFFFFBFLL | 0x2000000000000000LL;
  if ( !(unsigned int)sub_7BE280(56, 702, 0, 0) )
    goto LABEL_20;
  v13 = 0;
  v14 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(a1 + 121) & 0x40) != 0 )
    v13 = *(_QWORD *)(v14 + 208);
  v15 = *(_BYTE *)(v14 + 6);
  v16 = v15 >> 7;
  if ( dword_4F04C44 != -1 || (v15 & 6) != 0 || *(_BYTE *)(v14 + 4) == 12 )
    *(_BYTE *)(v14 + 6) |= 0x80u;
  v22 = v13;
  sub_65C7C0(a1);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6)
                                                           & 0x7F
                                                           | (v16 << 7);
  *(_QWORD *)(a1 + 200) = v12;
  *(_QWORD *)(a1 + 352) = sub_869D30();
  sub_65E230(&v23, a1, v22, (__int64)v27, v17);
  v18 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    if ( *(_BYTE *)(v18 + 80) == 3 )
    {
      v19 = *(_QWORD *)(v18 + 88);
      if ( *(_BYTE *)(v19 + 140) == 12 )
      {
        if ( *(_QWORD *)(v19 + 8) )
        {
          if ( (*(_BYTE *)(a1 + 127) & 0x10) != 0 )
          {
            *(_BYTE *)(v19 + 184) = 9;
          }
          else
          {
            v21 = *(_QWORD *)(a1 + 352);
            if ( v21 && *(_BYTE *)(v21 + 16) == 53 )
              *(_BYTE *)(*(_QWORD *)(v21 + 24) + 58LL) |= 2u;
          }
          *(_QWORD *)(*(_QWORD *)(v19 + 168) + 52LL) = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(*(_QWORD *)(v19 + 168) + 60LL) = unk_4F061D8;
LABEL_20:
          v18 = *(_QWORD *)a1;
        }
      }
    }
  }
  sub_86F690(v18);
  result = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  return result;
}
