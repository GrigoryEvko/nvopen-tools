// Function: sub_380CCA0
// Address: 0x380cca0
//
unsigned __int8 *__fastcall sub_380CCA0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int16 *v5; // rdx
  unsigned __int8 *v6; // r12
  __int64 v7; // r8
  __int16 v8; // ax
  __int64 v9; // rdx
  unsigned int v10; // r15d
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  int v17; // [rsp+18h] [rbp-38h]

  HIWORD(v10) = 0;
  v4 = sub_380AAE0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = *(__int16 **)(a2 + 48);
  v6 = (unsigned __int8 *)v4;
  v7 = *((_QWORD *)v5 + 1);
  v8 = *v5;
  v9 = *((_QWORD *)v6 + 6);
  LOWORD(v10) = v8;
  if ( *(_WORD *)v9 != v8 || *(_QWORD *)(v9 + 8) != v7 && !v8 )
  {
    v12 = *(_QWORD *)(a2 + 80);
    v13 = *(_QWORD *)(a1 + 8);
    v16 = v12;
    if ( v12 )
    {
      v14 = v13;
      v15 = v7;
      sub_B96E90((__int64)&v16, v12, 1);
      v13 = v14;
      v7 = v15;
    }
    v17 = *(_DWORD *)(a2 + 72);
    v6 = sub_33FAF80(v13, 233, (__int64)&v16, v10, v7, v13, a3);
    if ( v16 )
      sub_B91220((__int64)&v16, v16);
  }
  return v6;
}
