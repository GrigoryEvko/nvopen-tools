// Function: sub_2122060
// Address: 0x2122060
//
__int64 __fastcall sub_2122060(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r15d
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  __m128i *v9; // r11
  __int64 v10; // r9
  __int64 v11; // rdx
  char *v12; // rcx
  char v13; // si
  int v14; // ecx
  __int64 v15; // r14
  __int64 v17; // [rsp+0h] [rbp-80h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  __m128i *v19; // [rsp+8h] [rbp-78h]
  _QWORD v20[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  int v22; // [rsp+28h] [rbp-58h]
  __int64 v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  __int64 v25; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v23,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v24;
  v18 = v25;
  v7 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = (__m128i *)*a1;
  v20[0] = v7;
  v10 = v18;
  v20[1] = v11;
  v21 = v8;
  if ( v8 )
  {
    v17 = v18;
    v19 = v9;
    sub_1623A60((__int64)&v21, v8, 2);
    v10 = v17;
    v9 = v19;
  }
  v12 = *(char **)(a2 + 40);
  v22 = *(_DWORD *)(a2 + 64);
  v13 = *v12;
  v14 = 112;
  if ( v13 != 9 )
  {
    v14 = 113;
    if ( v13 != 10 )
    {
      v14 = 114;
      if ( v13 != 11 )
      {
        v14 = 115;
        if ( v13 != 12 )
        {
          v14 = 462;
          if ( v13 == 13 )
            v14 = 116;
        }
      }
    }
  }
  sub_20BE530((__int64)&v23, v9, a1[1], v14, v6, v10, a3, a4, a5, (__int64)v20, 1u, 0, (__int64)&v21, 0, 1);
  v15 = v23;
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v15;
}
