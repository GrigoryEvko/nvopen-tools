// Function: sub_64E550
// Address: 0x64e550
//
__int64 __fastcall sub_64E550(int a1, int a2)
{
  int v2; // r15d
  __m128i v3; // xmm0
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned int v18; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v19[0] = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] != 1 )
  {
    sub_7BE280(1, 40, 0, 0);
    v2 = 1;
    v3 = _mm_loadu_si128(xmmword_4F06660);
    v4 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v5 = _mm_loadu_si128(&xmmword_4F06660[3]);
    unk_4D04A10 = _mm_loadu_si128(&xmmword_4F06660[1]);
    unk_4D04A11 |= 0x20u;
    *(__m128i *)&qword_4D04A00 = v3;
    xmmword_4D04A20 = v4;
    unk_4D04A30 = v5;
    goto LABEL_3;
  }
  v16 = sub_87A430(qword_4D04A00);
  v7 = v16;
  if ( a2 )
  {
    if ( v16 )
    {
      v6 = (int)dword_4F04C5C;
      if ( *(_DWORD *)(v16 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
      {
        sub_685920(v19, v16, 8);
        goto LABEL_6;
      }
      v2 = 0;
LABEL_5:
      qword_4D04A08 = unk_4F077C8;
      v7 = sub_885AD0(12, &qword_4D04A00, v6, 1);
      sub_7296F0((unsigned int)dword_4F04C58, &v18);
      v8 = sub_726410();
      v9 = v18;
      *(_QWORD *)(v7 + 88) = v8;
      v10 = v8;
      sub_729730(v9);
      *(_BYTE *)(v10 + 120) = (32 * (a2 & 1)) | *(_BYTE *)(v10 + 120) & 0xDF;
      sub_730430(v10);
      sub_877D80(v10, v7);
      if ( v2 )
        goto LABEL_6;
      goto LABEL_9;
    }
    goto LABEL_19;
  }
  if ( !v16 )
  {
LABEL_19:
    v2 = 0;
LABEL_3:
    if ( a2 )
      v6 = (unsigned int)dword_4F04C5C;
    else
      v6 = (unsigned int)dword_4F04C58;
    goto LABEL_5;
  }
LABEL_9:
  if ( a1 )
  {
    v12 = v7;
    v13 = 3;
    sub_8756F0(3, v7, &dword_4F063F8, 0);
  }
  else if ( a2 )
  {
    v12 = v7;
    v13 = 1;
    sub_8756F0(1, v7, &dword_4F063F8, 0);
  }
  else
  {
    v12 = v7;
    v13 = 4;
    sub_8767A0(4, v7, &dword_4F063F8, 1);
    if ( (*(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFLL) == 0 )
      *(_QWORD *)(v7 + 48) = *(_QWORD *)&dword_4F063F8;
  }
  sub_7B8B50(v13, v12, v14, v15);
LABEL_6:
  *(_QWORD *)dword_4F07508 = v19[0];
  return *(_QWORD *)(v7 + 88);
}
