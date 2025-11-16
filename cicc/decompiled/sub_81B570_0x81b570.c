// Function: sub_81B570
// Address: 0x81b570
//
__int64 __fastcall sub_81B570(_BYTE *a1, const char *a2, char a3, char a4, int a5)
{
  _BYTE *v6; // r12
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  size_t v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  size_t v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v19; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v21; // [rsp+30h] [rbp-60h]
  __m128i v22; // [rsp+40h] [rbp-50h]
  __m128i v23; // [rsp+50h] [rbp-40h]

  v6 = a1;
  v19 = 0;
  if ( a1 )
    v6 = sub_819EB0(a1, &v19);
  v8 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v9 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v10 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v20[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v21 = v8;
  v22 = v9;
  v23 = v10;
  v20[1] = *(_QWORD *)&dword_4F077C8;
  v11 = strlen(a2);
  v12 = sub_87A510(a2, v11, v20);
  v13 = v12;
  if ( v12 )
  {
    if ( !sub_81A4B0((__int64)v6, v19 - 1, *(_BYTE **)(*(_QWORD *)(v12 + 88) + 16LL)) )
      sub_685220(0x53Au, (__int64)a2);
  }
  else
  {
    v15 = strlen(a2);
    v13 = sub_885B80(a2, v15, 1, 0xFFFFFFFFLL);
    v16 = sub_823970(24);
    sub_81B550((unsigned __int8 *)v16);
    *(_QWORD *)(v13 + 88) = v16;
    if ( a5 )
    {
      *(_BYTE *)v16 &= ~1u;
      v17 = sub_823970(24);
      *(_WORD *)(v17 + 16) = 0;
      *(_QWORD *)(v17 + 8) = 0;
      *(_QWORD *)(v16 + 8) = v17;
      *(_QWORD *)v17 = byte_3F871B3;
      *(_BYTE *)(*(_QWORD *)(v16 + 8) + 16LL) = 1;
    }
    else
    {
      *(_BYTE *)v16 |= 1u;
      *(_QWORD *)(v16 + 8) = 0;
    }
    *(_QWORD *)(v16 + 16) = v6;
    *(_BYTE *)v16 = *(_BYTE *)v16 & 0xE9 | (2 * (a3 & 1)) | 0x10 | (4 * (a4 & 1));
  }
  return v13;
}
