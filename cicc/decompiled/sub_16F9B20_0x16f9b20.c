// Function: sub_16F9B20
// Address: 0x16f9b20
//
__int64 __fastcall sub_16F9B20(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // r13d
  _BYTE *v4; // rsi
  __int64 v5; // rax
  _BYTE *v6; // rcx
  __int64 v7; // rax
  __m128i v8; // xmm0
  unsigned __int64 v9; // rbx
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // r8d
  int v14; // r9d
  _QWORD *v15; // rdi
  __int64 result; // rax
  __m128i v17; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]
  _QWORD v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_DWORD *)(a1 + 60);
  sub_16F7930(a1, 1u);
  v4 = *(_BYTE **)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v4 != (_BYTE *)v5 )
  {
    if ( !(unsigned __int8)sub_16F7940(a1, v4) )
    {
      v6 = *(_BYTE **)(a1 + 40);
      if ( *v6 != 60 )
      {
        v5 = sub_16F7770(a1, (char *)sub_16F6460, 0, (__int64)v6);
        *(_QWORD *)(a1 + 40) = v5;
        goto LABEL_5;
      }
      sub_16F7930(a1, 1u);
      sub_16F77F0(a1);
      result = sub_16F78D0(a1, 0x3Eu);
      if ( !(_BYTE)result )
        return result;
    }
    v5 = *(_QWORD *)(a1 + 40);
  }
LABEL_5:
  v17.m128i_i64[0] = v2;
  v18 = v20;
  v19 = 0;
  LOBYTE(v20[0]) = 0;
  v17.m128i_i64[1] = v5 - v2;
  v7 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v8 = _mm_loadu_si128(&v17);
  v9 = v7;
  *(_QWORD *)v7 = 0;
  v10 = v19;
  *(_QWORD *)(v7 + 8) = 0;
  *(__m128i *)(v7 + 24) = v8;
  *(_DWORD *)(v7 + 16) = 22;
  *(_QWORD *)(v7 + 40) = v7 + 56;
  sub_16F6740((__int64 *)(v7 + 40), v20, (__int64)v20 + v10);
  *(_QWORD *)(v9 + 8) = a1 + 184;
  v11 = *(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v9 = v11 | *(_QWORD *)v9 & 7LL;
  *(_QWORD *)(v11 + 8) = v9;
  v12 = *(_QWORD *)(a1 + 184) & 7LL | v9;
  *(_QWORD *)(a1 + 184) = v12;
  sub_16F79B0(a1, v12 & 0xFFFFFFFFFFFFFFF8LL, v3, 0, v13, v14);
  v15 = v18;
  *(_BYTE *)(a1 + 73) = 0;
  if ( v15 != v20 )
    j_j___libc_free_0(v15, v20[0] + 1LL);
  return 1;
}
