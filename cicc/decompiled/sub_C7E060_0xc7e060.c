// Function: sub_C7E060
// Address: 0xc7e060
//
__int64 __fastcall sub_C7E060(__int64 a1, __int64 *a2)
{
  char v2; // dl
  char v3; // al
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v9; // r14
  __m128i v10; // xmm1
  int v11; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  char v13; // [rsp+18h] [rbp-68h]
  __m128i v14[2]; // [rsp+20h] [rbp-60h] BYREF
  __m128i v15; // [rsp+40h] [rbp-40h] BYREF
  char v16; // [rsp+50h] [rbp-30h]

  sub_C83520(&v12, a2, 0, 0);
  v2 = v13 & 1;
  v3 = (2 * (v13 & 1)) | v13 & 0xFD;
  v13 = v3;
  if ( v2 )
  {
    v13 = v3 & 0xFD;
    v4 = v12;
    v12 = 0;
    v15.m128i_i64[0] = v4 | 1;
    v5 = sub_C64300(v15.m128i_i64, a2);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v5;
    v6 = v15.m128i_i64[0];
    *(_QWORD *)(a1 + 8) = v7;
    if ( (v6 & 1) != 0 || (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v15, (__int64)a2);
  }
  else
  {
    v11 = v12;
    sub_C7DE70((__int64)&v15, (__int64 *)(unsigned int)v12, (__int64)a2);
    if ( (v16 & 1) != 0 )
    {
      v14[0] = _mm_loadu_si128(&v15);
      sub_C83820(&v11);
      v10 = _mm_loadu_si128(v14);
      *(_BYTE *)(a1 + 16) |= 1u;
      *(__m128i *)a1 = v10;
    }
    else
    {
      v9 = v15.m128i_i64[0];
      sub_C83820(&v11);
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_QWORD *)a1 = v9;
    }
  }
  if ( (v13 & 2) != 0 )
    sub_C0EC50(&v12);
  if ( (v13 & 1) == 0 || !v12 )
    return a1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  return a1;
}
