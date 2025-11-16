// Function: sub_18E4B00
// Address: 0x18e4b00
//
__int64 __fastcall sub_18E4B00(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        _QWORD *a4,
        __int64 *a5,
        __int64 *a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rbx
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+18h] [rbp-38h]

  v11 = a3;
  v12 = (_QWORD *)sub_16498A0(a2);
  v13 = sub_1643360(v12);
  v14 = 0;
  if ( *(char *)(a2 + 23) < 0 )
    v14 = sub_1648A40(a2);
  v15 = v14 + 16 * v11;
  v16 = *(_QWORD *)v15;
  v17 = *(unsigned int *)(v15 + 8);
  v18 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v19 = *(unsigned int *)(v15 + 12);
  if ( *(_QWORD *)v16 != 5 )
    return 0;
  if ( *(_DWORD *)(v16 + 16) != 1734962273 )
    return 0;
  if ( *(_BYTE *)(v16 + 20) != 110 )
    return 0;
  v21 = a4;
  v26 = 24 * v17;
  v29 = (_QWORD *)(a2 + 24 * v17 - 24 * v18);
  *v21 = *v29;
  v22 = sub_146F1B0(*a1, v29[3]);
  *a5 = v22;
  v23 = sub_1483B20((_QWORD *)*a1, v22, v13, a7, a8);
  *a5 = v23;
  if ( *(_WORD *)(v23 + 24) )
    return 0;
  v24 = *a1;
  if ( 24 * v19 - 72 == v26 )
    v25 = sub_146F1B0(v24, v29[6]);
  else
    v25 = sub_145CF80(v24, v13, 0, 0);
  *a6 = v25;
  *a6 = sub_1483B20((_QWORD *)*a1, v25, v13, a7, a8);
  return 1;
}
