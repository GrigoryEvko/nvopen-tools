// Function: sub_21229C0
// Address: 0x21229c0
//
__int64 __fastcall sub_21229C0(__int64 *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r15d
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  __m128i *v9; // r11
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __m128i v13; // xmm0
  char *v14; // rcx
  char v15; // si
  int v16; // ecx
  __int64 v17; // r14
  __int64 v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+8h] [rbp-88h]
  __m128i *v21; // [rsp+8h] [rbp-88h]
  __int64 v22; // [rsp+10h] [rbp-80h] BYREF
  int v23; // [rsp+18h] [rbp-78h]
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v25; // [rsp+30h] [rbp-60h]
  __int64 v26; // [rsp+40h] [rbp-50h] BYREF
  __int64 v27; // [rsp+48h] [rbp-48h]
  __int64 v28; // [rsp+50h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v26,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v27;
  v20 = v28;
  v7 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = (__m128i *)*a1;
  v24[0] = v7;
  v10 = *(_QWORD *)(a2 + 32);
  v24[1] = v11;
  v12 = v20;
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v22 = v8;
  v25 = v13;
  if ( v8 )
  {
    v19 = v20;
    v21 = v9;
    sub_1623A60((__int64)&v22, v8, 2);
    v12 = v19;
    v9 = v21;
  }
  v14 = *(char **)(a2 + 40);
  v23 = *(_DWORD *)(a2 + 64);
  v15 = *v14;
  v16 = 82;
  if ( v15 != 9 )
  {
    v16 = 83;
    if ( v15 != 10 )
    {
      v16 = 84;
      if ( v15 != 11 )
      {
        v16 = 85;
        if ( v15 != 12 )
        {
          v16 = 462;
          if ( v15 == 13 )
            v16 = 86;
        }
      }
    }
  }
  sub_20BE530((__int64)&v26, v9, a1[1], v16, v6, v12, v13, a4, a5, (__int64)v24, 2u, 0, (__int64)&v22, 0, 1);
  v17 = v26;
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v17;
}
