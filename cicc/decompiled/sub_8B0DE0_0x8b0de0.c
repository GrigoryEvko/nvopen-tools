// Function: sub_8B0DE0
// Address: 0x8b0de0
//
__int64 __fastcall sub_8B0DE0(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned int v6; // esi
  __int64 v7; // rdi
  __m128i v8; // xmm1
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21; // rax
  unsigned int v22; // eax
  unsigned __int16 v24; // [rsp+1Eh] [rbp-432h]
  int v25; // [rsp+20h] [rbp-430h]
  unsigned int v26; // [rsp+24h] [rbp-42Ch]
  unsigned __int16 v27; // [rsp+28h] [rbp-428h]
  __int16 v28; // [rsp+2Ah] [rbp-426h]
  int v29; // [rsp+2Ch] [rbp-424h]
  int v30[2]; // [rsp+30h] [rbp-420h] BYREF
  __int64 v31; // [rsp+38h] [rbp-418h]
  _QWORD v32[60]; // [rsp+40h] [rbp-410h] BYREF
  _BYTE v33[44]; // [rsp+220h] [rbp-230h] BYREF
  _BOOL4 v34; // [rsp+24Ch] [rbp-204h]
  __int64 v35; // [rsp+26Ch] [rbp-1E4h]
  __m128i v36; // [rsp+318h] [rbp-138h]
  __m128i v37; // [rsp+328h] [rbp-128h]

  v3 = a1;
  v4 = (__int64)a3;
  v26 = dword_4F063F8;
  v27 = word_4F063FC[0];
  v25 = dword_4F07508[0];
  v28 = dword_4F07508[1];
  v29 = dword_4F061D8;
  v24 = word_4F061DC[0];
  if ( dword_4F04C44 != -1
    || (v21 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v21 + 6) & 6) != 0)
    || *(_BYTE *)(v21 + 4) == 12
    || (v22 = sub_89A370(a3), v4 = (__int64)a3, v3 = a1, (v6 = v22) != 0) )
  {
    v6 = 2052;
  }
  if ( dword_4F60178 == unk_4D042F0 )
  {
    sub_6851C0(0x3FCu, dword_4F07508);
    result = *(_QWORD *)(*(_QWORD *)(sub_87F550() + 88) + 104LL);
  }
  else
  {
    v7 = a2[3].m128i_i64[0];
    ++dword_4F60178;
    sub_865840(v7, 0, 0, 0, v3, v4, v6);
    sub_7BC160((__int64)a2[1].m128i_i64);
    memset(v32, 0, 0x1D8u);
    v32[19] = v32;
    v32[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v32[22]) |= 1u;
    sub_891F00((__int64)v33, (__int64)v32);
    v8 = _mm_loadu_si128(a2 + 2);
    v9 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6);
    v36 = _mm_loadu_si128(a2 + 1);
    v37 = v8;
    v30[1] = 0;
    v31 = 0;
    v34 = (v9 & 2) != 0;
    v35 = 0x100000001LL;
    v30[0] = *(_DWORD *)(*(_QWORD *)(a2[4].m128i_i64[0] + 104) + 132LL);
    v14 = sub_8B06F0((__int64)v33, v30, 1u);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50((unsigned __int64)v33, (unsigned int *)v30, v10, v11, v12, v13);
    sub_7B8B50((unsigned __int64)v33, (unsigned int *)v30, v10, v11, v12, v13);
    sub_863FE0((__int64)v33, (__int64)v30, v15, v16, v17, v18);
    v19 = v14[4].m128i_i64[0];
    --dword_4F60178;
    result = *(_QWORD *)(v19 + 104);
  }
  dword_4F07508[0] = v25;
  word_4F063FC[0] = v27;
  LOWORD(dword_4F07508[1]) = v28;
  word_4F061DC[0] = v24;
  dword_4F063F8 = v26;
  dword_4F061D8 = v29;
  return result;
}
