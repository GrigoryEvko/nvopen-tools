// Function: sub_21BED50
// Address: 0x21bed50
//
unsigned __int64 __fastcall sub_21BED50(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 result; // rax
  __int128 v15; // [rsp-10h] [rbp-50h]
  unsigned __int64 v16; // [rsp-10h] [rbp-50h]
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+10h] [rbp-30h] BYREF
  int v20; // [rsp+18h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 72);
  v5 = *(_QWORD **)(a1 + 272);
  v6 = *(__int64 **)(*(_QWORD *)(v3 + 40) + 32LL);
  v7 = *v6;
  v8 = v6[1];
  v19 = v4;
  if ( v4 )
  {
    v17 = v7;
    v18 = v8;
    sub_1623A60((__int64)&v19, v4, 2);
    v7 = v17;
    v8 = v18;
  }
  *((_QWORD *)&v15 + 1) = v8;
  *(_QWORD *)&v15 = v7;
  v20 = *(_DWORD *)(a2 + 64);
  v9 = sub_1D2CC80(v5, 4889, (__int64)&v19, 6, 0, v8, v15);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v9);
  sub_1D49010(v9);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v10, v11, v12, v13);
  result = v16;
  if ( v19 )
    return sub_161E7C0((__int64)&v19, v19);
  return result;
}
