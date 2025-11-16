// Function: sub_2123370
// Address: 0x2123370
//
__int64 __fastcall sub_2123370(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  const void **v6; // r15
  unsigned int v7; // r14d
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  __m128i *v10; // r11
  __int64 v11; // rdx
  char *v12; // rcx
  char v13; // si
  int v14; // ecx
  __int64 v15; // r14
  __int64 v17; // rsi
  __int64 *v18; // r12
  __int128 *v19; // rcx
  __m128i *v20; // [rsp+0h] [rbp-80h]
  __int128 *v21; // [rsp+8h] [rbp-78h]
  _QWORD v22[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+20h] [rbp-60h] BYREF
  int v24; // [rsp+28h] [rbp-58h]
  __int64 v25; // [rsp+30h] [rbp-50h] BYREF
  __int64 v26; // [rsp+38h] [rbp-48h]
  const void **v27; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v25,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = v27;
  v7 = (unsigned __int8)v26;
  if ( **(_BYTE **)(a2 + 40) == 8 )
  {
    v17 = *(_QWORD *)(a2 + 72);
    v18 = (__int64 *)a1[1];
    v19 = *(__int128 **)(a2 + 32);
    v25 = v17;
    if ( v17 )
    {
      v21 = v19;
      sub_1623A60((__int64)&v25, v17, 2);
      v19 = v21;
    }
    LODWORD(v26) = *(_DWORD *)(a2 + 64);
    v15 = sub_1D309E0(
            v18,
            161,
            (__int64)&v25,
            v7,
            v6,
            0,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            *v19);
    if ( v25 )
      sub_161E7C0((__int64)&v25, v25);
  }
  else
  {
    v8 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
    v9 = *(_QWORD *)(a2 + 72);
    v10 = (__m128i *)*a1;
    v22[0] = v8;
    v22[1] = v11;
    v23 = v9;
    if ( v9 )
    {
      v20 = v10;
      sub_1623A60((__int64)&v23, v9, 2);
      v10 = v20;
    }
    v12 = *(char **)(a2 + 40);
    v24 = *(_DWORD *)(a2 + 64);
    v13 = *v12;
    v14 = 174;
    if ( v13 != 9 )
    {
      v14 = 175;
      if ( v13 != 10 )
      {
        v14 = 176;
        if ( v13 != 11 )
        {
          v14 = 177;
          if ( v13 != 12 )
          {
            v14 = 462;
            if ( v13 == 13 )
              v14 = 178;
          }
        }
      }
    }
    sub_20BE530((__int64)&v25, v10, a1[1], v14, v7, (__int64)v6, a3, a4, a5, (__int64)v22, 1u, 0, (__int64)&v23, 0, 1);
    v15 = v25;
    if ( v23 )
      sub_161E7C0((__int64)&v23, v23);
  }
  return v15;
}
