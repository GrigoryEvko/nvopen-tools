// Function: sub_8B2D80
// Address: 0x8b2d80
//
__int64 __fastcall sub_8B2D80(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // r14
  _QWORD *v8; // r15
  __m128i *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 v14; // rdi
  __int64 *v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  _QWORD *v20; // r14
  _QWORD *v21; // rax
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 *v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __m128i *v25; // [rsp+10h] [rbp-40h] BYREF
  __m128i *v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = **(_QWORD **)(sub_8794A0(a2) + 32);
  v5 = sub_8794A0(a1);
  result = 1;
  if ( (*(_BYTE *)(v5 + 160) & 2) == 0 )
  {
    v22 = v5;
    v24 = *a1;
    v7 = **(_QWORD **)(v5 + 32);
    v8 = sub_8972D0(v7);
    *(_BYTE *)(v8[11] + 160LL) = *(_BYTE *)(v22 + 160) & 0x20 | *(_BYTE *)(v8[11] + 160LL) & 0xDF;
    *(_QWORD *)(v8[11] + 104LL) = a1;
    sub_865900((__int64)v8);
    v25 = (__m128i *)sub_896D70(0, v7, 0);
    v23 = sub_8AF060((__int64)v8, &v25);
    v26[0] = (__m128i *)sub_896D70(0, v4, 0);
    v9 = v26[0];
    v26[0] = sub_8A3C00(v7, (__int64)v26[0]->m128i_i64, 1, (__int64 *)(v24 + 48));
    if ( v26[0] )
    {
      v14 = (__int64)v8;
      v15 = sub_8AF060((__int64)v8, v26);
      sub_864110(v14, (__int64)v26, v16, v17, v18, v19);
      result = 0;
      if ( v15 )
      {
        v20 = sub_88D6D0(v7, v23[11]);
        *(_QWORD *)(v20[11] + 104LL) = a1;
        v21 = sub_88D6D0(v4, v15[11]);
        *(_QWORD *)(v21[11] + 104LL) = a2;
        return (unsigned int)sub_8B2440((unsigned __int64)v20, (unsigned __int64)v21, 2, 1u) >> 31;
      }
    }
    else
    {
      sub_864110(v7, (__int64)v9, v10, v11, v12, v13);
      return 0;
    }
  }
  return result;
}
