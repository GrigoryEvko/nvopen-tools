// Function: sub_1E75850
// Address: 0x1e75850
//
__int64 __fastcall sub_1E75850(__int64 a1, __int8 *a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int8 v12; // dl
  _BYTE v13[4]; // [rsp+18h] [rbp-78h] BYREF
  __int64 v14; // [rsp+1Ch] [rbp-74h]
  _BYTE v15[4]; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v16; // [rsp+28h] [rbp-68h]
  __m128i v17; // [rsp+30h] [rbp-60h] BYREF
  __m128i v18; // [rsp+40h] [rbp-50h]
  __m128i v19; // [rsp+50h] [rbp-40h]

  v2 = a1 + 512;
  result = sub_1E756D0(a1 + 512);
  if ( result )
  {
    *a2 = 0;
  }
  else
  {
    result = sub_1E756D0(a1 + 144);
    if ( result )
    {
      *a2 = 1;
    }
    else
    {
      v13[0] = 0;
      v14 = 0;
      sub_1E73790(a1, (__int64)v13, 0, v2, a1 + 144);
      v15[0] = 0;
      v16 = 0;
      sub_1E73790(a1, (__int64)v15, 0, a1 + 144, v2);
      v4 = *(_QWORD *)(a1 + 944);
      if ( !v4 || (*(_BYTE *)(v4 + 229) & 4) != 0 || v13[0] != *(_BYTE *)(a1 + 928) || *(_QWORD *)(a1 + 932) != v14 )
      {
        v5 = *(_QWORD *)(a1 + 128);
        *(_WORD *)(a1 + 964) = 0;
        *(_BYTE *)(a1 + 928) = 0;
        *(_QWORD *)(a1 + 932) = 0;
        *(_QWORD *)(a1 + 944) = 0;
        *(_QWORD *)(a1 + 952) = 0;
        *(_DWORD *)(a1 + 960) = 0;
        *(_QWORD *)(a1 + 968) = 0;
        sub_1E74440((__int64 *)a1, v2, (__int64 *)v13, v5 + 3776, a1 + 928);
      }
      v6 = *(_QWORD *)(a1 + 896);
      if ( !v6 || (*(_BYTE *)(v6 + 229) & 4) != 0 || *(_BYTE *)(a1 + 880) != v15[0] || *(_QWORD *)(a1 + 884) != v16 )
      {
        *(_BYTE *)(a1 + 880) = 0;
        *(_WORD *)(a1 + 916) = 0;
        v7 = *(_QWORD *)(a1 + 128);
        *(_QWORD *)(a1 + 884) = 0;
        *(_QWORD *)(a1 + 896) = 0;
        *(_QWORD *)(a1 + 904) = 0;
        *(_DWORD *)(a1 + 912) = 0;
        *(_QWORD *)(a1 + 920) = 0;
        sub_1E74440((__int64 *)a1, a1 + 144, (__int64 *)v15, v7 + 3288, a1 + 880);
      }
      v8 = _mm_loadu_si128((const __m128i *)(a1 + 928));
      v9 = *(_QWORD *)a1;
      v10 = _mm_loadu_si128((const __m128i *)(a1 + 944));
      v11 = _mm_loadu_si128((const __m128i *)(a1 + 960));
      *(_BYTE *)(a1 + 904) = 0;
      v17 = v8;
      v18 = v10;
      v19 = v11;
      (*(void (__fastcall **)(__int64, __m128i *, __int64, _QWORD))(v9 + 136))(a1, &v17, a1 + 880, 0);
      if ( *(_BYTE *)(a1 + 904) )
      {
        result = *(_QWORD *)(a1 + 896);
        v12 = *(_BYTE *)(a1 + 905);
      }
      else
      {
        v12 = v18.m128i_i8[9];
        result = v18.m128i_i64[0];
      }
      *a2 = v12;
    }
  }
  return result;
}
