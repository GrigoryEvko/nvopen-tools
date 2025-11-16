// Function: sub_65B760
// Address: 0x65b760
//
__int64 __fastcall sub_65B760(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // r12
  __int64 *v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r15
  char v11; // al
  _BYTE *v12; // rdx
  __int64 result; // rax
  __int64 v14; // rax
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __int64 v18; // r15
  __int64 v19; // r14
  char v20; // bl
  __int64 v21; // rax
  char v22[4]; // [rsp+14h] [rbp-7Ch] BYREF
  __int64 v23; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v25; // [rsp+30h] [rbp-60h]
  __m128i v26; // [rsp+40h] [rbp-50h]
  __m128i v27; // [rsp+50h] [rbp-40h]

  v5 = a2;
  sub_7B8B50(a1, a2, a3, a4);
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] == 1 )
      goto LABEL_3;
LABEL_11:
    sub_6851D0(40);
    sub_854B40();
    goto LABEL_8;
  }
  if ( (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(0x40000, 0) )
    goto LABEL_11;
LABEL_3:
  v6 = (__int64 *)1;
  v7 = 0x40000;
  v23 = *(_QWORD *)&dword_4F063F8;
  v8 = sub_7BF130(0x40000, 1, v22);
  v10 = v8;
  if ( v8 )
  {
    v11 = *(_BYTE *)(v8 + 80);
    if ( v11 == 6 )
    {
LABEL_14:
      v14 = *(_QWORD *)(v10 + 88);
LABEL_16:
      while ( *(_BYTE *)(v14 + 140) == 12 )
        v14 = *(_QWORD *)(v14 + 160);
      v15 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v16 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v17 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v24[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v25 = v15;
      v25.m128i_i8[2] = v15.m128i_i8[2] | 2;
      v26 = v16;
      v24[1] = unk_4F077C8;
      v12 = *(_BYTE **)(v14 + 176);
      v26.m128i_i64[0] = v14;
      v27 = v17;
      if ( (*v12 & 1) != 0 )
      {
        v18 = *(_QWORD *)(v14 + 168);
        if ( (*(_BYTE *)(v14 + 161) & 0x10) != 0 )
          v18 = *(_QWORD *)(v18 + 96);
        if ( v18 )
        {
          v19 = 0;
          v20 = 1;
          do
          {
            v6 = *(__int64 **)v18;
            v7 = (__int64)v24;
            v21 = sub_65A3F0((__int64)v24, *(__int64 **)v18, a1, v5, v19);
            if ( v21 )
            {
              v19 = v21;
              v12 = (_BYTE *)(*(_BYTE *)(v21 + 41) & 0xFC);
              *(_BYTE *)(v21 + 41) = *(_BYTE *)(v21 + 41) & 0xFC | v20 | 2;
              v20 = 0;
            }
            v18 = *(_QWORD *)(v18 + 120);
          }
          while ( v18 );
        }
      }
      goto LABEL_7;
    }
    if ( v11 == 3 )
    {
      v7 = *(_QWORD *)(v10 + 88);
      if ( (unsigned int)sub_8D2870(v7) )
      {
        v14 = *(_QWORD *)(v10 + 88);
        if ( *(_BYTE *)(v10 + 80) != 6 )
          goto LABEL_16;
        goto LABEL_14;
      }
    }
    v6 = &v23;
    v7 = 3179;
    sub_6854C0(3179, &v23, v10);
  }
  else
  {
    v6 = &v23;
    v7 = 20;
    sub_6851A0(20, &v23, *(_QWORD *)(qword_4D04A00 + 8));
  }
LABEL_7:
  sub_7B8B50(v7, v6, v12, v9);
LABEL_8:
  result = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  return result;
}
