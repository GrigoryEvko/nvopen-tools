// Function: sub_21221B0
// Address: 0x21221b0
//
__int64 __fastcall sub_21221B0(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __m128i *v13; // r11
  __int64 v14; // r9
  __int64 v15; // rdx
  char *v16; // rcx
  char v17; // si
  int v18; // ecx
  __int64 v19; // r14
  __int64 v21; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v22; // [rsp+8h] [rbp-98h]
  __m128i *v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h] BYREF
  int v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+48h] [rbp-58h]
  unsigned __int64 v29; // [rsp+50h] [rbp-50h]
  __int64 v30; // [rsp+58h] [rbp-48h]
  unsigned __int64 v31; // [rsp+60h] [rbp-40h]
  __int64 v32; // [rsp+68h] [rbp-38h]

  sub_1F40D10(
    (__int64)&v27,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v28;
  v22 = v29;
  v27 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = *(_QWORD *)(a2 + 32);
  v28 = v8;
  v29 = sub_2120330((__int64)a1, *(_QWORD *)(v7 + 40), *(_QWORD *)(v7 + 48));
  v9 = *(_QWORD *)(a2 + 32);
  v30 = v10;
  v11 = sub_2120330((__int64)a1, *(_QWORD *)(v9 + 80), *(_QWORD *)(v9 + 88));
  v12 = *(_QWORD *)(a2 + 72);
  v13 = (__m128i *)*a1;
  v31 = v11;
  v14 = v22;
  v32 = v15;
  v24 = v12;
  if ( v12 )
  {
    v21 = v22;
    v23 = v13;
    sub_1623A60((__int64)&v24, v12, 2);
    v14 = v21;
    v13 = v23;
  }
  v16 = *(char **)(a2 + 40);
  v25 = *(_DWORD *)(a2 + 64);
  v17 = *v16;
  v18 = 77;
  if ( v17 != 9 )
  {
    v18 = 78;
    if ( v17 != 10 )
    {
      v18 = 79;
      if ( v17 != 11 )
      {
        v18 = 80;
        if ( v17 != 12 )
        {
          v18 = 462;
          if ( v17 == 13 )
            v18 = 81;
        }
      }
    }
  }
  sub_20BE530((__int64)&v26, v13, a1[1], v18, v6, v14, a3, a4, a5, (__int64)&v27, 3u, 0, (__int64)&v24, 0, 1);
  v19 = v26;
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v19;
}
