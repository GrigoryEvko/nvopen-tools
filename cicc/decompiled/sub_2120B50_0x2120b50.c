// Function: sub_2120B50
// Address: 0x2120b50
//
__int64 __fastcall sub_2120B50(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __m128i *v11; // r11
  __int64 v12; // r9
  __int64 v13; // rdx
  char *v14; // rcx
  char v15; // si
  int v16; // ecx
  __int64 v17; // r14
  __m128i *v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+10h] [rbp-80h] BYREF
  int v22; // [rsp+18h] [rbp-78h]
  _QWORD v23[4]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v24; // [rsp+40h] [rbp-50h] BYREF
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v24,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v25;
  v20 = v26;
  v23[0] = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = *(_QWORD *)(a2 + 32);
  v23[1] = v8;
  v9 = sub_2120330((__int64)a1, *(_QWORD *)(v7 + 40), *(_QWORD *)(v7 + 48));
  v10 = *(_QWORD *)(a2 + 72);
  v11 = (__m128i *)*a1;
  v23[2] = v9;
  v12 = v20;
  v23[3] = v13;
  v21 = v10;
  if ( v10 )
  {
    v19 = v11;
    sub_1623A60((__int64)&v21, v10, 2);
    v11 = v19;
    v12 = v20;
  }
  v14 = *(char **)(a2 + 40);
  v22 = *(_DWORD *)(a2 + 64);
  v15 = *v14;
  v16 = 209;
  if ( v15 != 9 )
  {
    v16 = 210;
    if ( v15 != 10 )
    {
      v16 = 211;
      if ( v15 != 11 )
      {
        v16 = 212;
        if ( v15 != 12 )
        {
          v16 = 462;
          if ( v15 == 13 )
            v16 = 213;
        }
      }
    }
  }
  sub_20BE530((__int64)&v24, v11, a1[1], v16, v6, v12, a3, a4, a5, (__int64)v23, 2u, 0, (__int64)&v21, 0, 1);
  v17 = v24;
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v17;
}
