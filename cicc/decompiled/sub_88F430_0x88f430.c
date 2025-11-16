// Function: sub_88F430
// Address: 0x88f430
//
__int64 __fastcall sub_88F430(__int64 a1, __m128i *a2)
{
  const __m128i *v2; // r12
  __int64 v4; // rax
  __m128i *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  _UNKNOWN *__ptr32 *v8; // r8
  unsigned int v9; // [rsp+Ch] [rbp-34h]
  int v10; // [rsp+1Ch] [rbp-24h] BYREF
  const __m128i *v11; // [rsp+20h] [rbp-20h] BYREF
  __m128i *v12; // [rsp+28h] [rbp-18h] BYREF

  v2 = a2;
  if ( (unsigned int)sub_72E9D0(a2, &v11, &v10) )
    v2 = v11;
  if ( v2[10].m128i_i8[13] != 12 )
    return 0;
  if ( v2[11].m128i_i8[0] )
    return 0;
  v4 = sub_8D4940(*(_QWORD *)(a1 + 128));
  if ( (unsigned int)sub_8D3EA0(v4) )
    return 0;
  v12 = (__m128i *)sub_724DC0();
  sub_72A510(v2, v12);
  v5 = v12;
  v12[8].m128i_i64[0] = *(_QWORD *)(a1 + 128);
  v9 = sub_73A2C0(a1, (__int64)v5, v6, v7, v8);
  sub_724E30((__int64)&v12);
  return v9;
}
