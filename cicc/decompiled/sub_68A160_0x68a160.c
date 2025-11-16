// Function: sub_68A160
// Address: 0x68a160
//
__int64 __fastcall sub_68A160(char *src)
{
  __m128i v1; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  size_t v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12
  char v8; // al
  char v10; // al
  __int64 v11; // rax
  _QWORD v12[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v13; // [rsp+10h] [rbp-50h]
  __m128i v14; // [rsp+20h] [rbp-40h]
  __m128i v15; // [rsp+30h] [rbp-30h]

  v1 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v2 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v3 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v12[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v13 = v1;
  v14 = v2;
  v15 = v3;
  v12[1] = *(_QWORD *)&dword_4F077C8;
  v4 = strlen(src);
  sub_878540(src, v4);
  v5 = sub_7D4600(unk_4F07288, v12, 2621440);
  v6 = v12[0];
  v7 = v5;
  v8 = *(_BYTE *)(v12[0] + 73LL);
  if ( (v8 & 0x20) == 0 )
    goto LABEL_8;
  if ( !unk_4D03FE8 )
  {
    if ( !(unsigned int)sub_889670() )
      goto LABEL_8;
    v6 = v12[0];
    goto LABEL_16;
  }
  if ( (v8 & 0x40) == 0 )
  {
LABEL_16:
    sub_889E70(v6);
    v7 = sub_7D4600(unk_4F07288, v12, 2621440);
    goto LABEL_8;
  }
  if ( !v7 )
    return v7;
  while ( 1 )
  {
    v10 = *(_BYTE *)(v7 + 80);
    if ( v10 == 17 )
    {
      v7 = *(_QWORD *)(v7 + 88);
      goto LABEL_7;
    }
    if ( v10 == 11 )
    {
      v11 = *(_QWORD *)(v7 + 88);
      if ( !*(_BYTE *)(v11 + 174) )
      {
        if ( *(_WORD *)(v11 + 176) )
          return v7;
      }
    }
LABEL_7:
    v7 = *(_QWORD *)(v7 + 8);
LABEL_8:
    if ( !v7 )
      return v7;
  }
}
