// Function: sub_65A3F0
// Address: 0x65a3f0
//
__int64 __fastcall sub_65A3F0(__int64 a1, __int64 *a2, __int64 a3, char a4, __int64 a5)
{
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int16 v13; // dx
  __int16 v14; // ax
  __int64 v15; // rax
  __int64 v16; // rdi
  char v17; // al
  __int64 v20; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-70h] BYREF
  __m128i v22; // [rsp+40h] [rbp-60h]
  __m128i v23; // [rsp+50h] [rbp-50h]
  __m128i v24; // [rsp+60h] [rbp-40h]

  v6 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v20 = qword_4D04A08;
  v8 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v9 = *a2;
  v21[1] = qword_4D04A08;
  v22 = v6;
  v21[0] = v9;
  v23 = v7;
  v24 = v8;
  v10 = sub_7CFB70(v21, 0);
  if ( a2 != (__int64 *)v10 || (v11 = 0, a3) )
  {
    if ( v10 )
    {
      v11 = 0;
      if ( a2 == (__int64 *)v10 )
        sub_686A30(8, 3181, &v20, a2 + 6);
      else
        sub_686C60(3180, &v20, v10, a2);
    }
    else
    {
      v11 = sub_6506C0((__int64)a2, &v20, dword_4F04C64);
      v13 = *(_WORD *)(v11 + 40);
      *(_QWORD *)(v11 + 48) = *(_QWORD *)(a1 + 32);
      *(_BYTE *)(v11 + 42) = a4;
      LOBYTE(v14) = *(_BYTE *)(a1 + 18) & 2;
      HIBYTE(v14) = 4;
      *(_WORD *)(v11 + 40) = v13 & 0xFBFD | v14;
      sub_876960(a2, &v20, v11, a5);
      v15 = sub_885A40(a2, 1, v21, (unsigned int)dword_4F04C64, 0);
      v16 = v15;
      if ( a3 )
      {
        *(_BYTE *)(v15 + 96) = *(_BYTE *)(v15 + 96) & 0xFC | a4 & 3;
        sub_877E20(v15, 0, a3);
      }
      else
      {
        v17 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
        if ( (unsigned __int8)(v17 - 3) <= 1u || !v17 )
          sub_877E90(v16, 0);
      }
    }
  }
  return v11;
}
