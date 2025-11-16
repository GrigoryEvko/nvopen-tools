// Function: sub_39CBE70
// Address: 0x39cbe70
//
__m128i *__fastcall sub_39CBE70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *v6; // r12
  unsigned __int8 *v8; // r15
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int8 *v11; // rax
  int v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  int v16; // r9d
  __int64 v17; // rdx
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  void *v20; // rcx
  size_t v21; // rdx
  size_t v22; // r8
  unsigned __int8 *v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  int v26[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = (__m128i *)sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( v6 )
    return v6;
  v8 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v24 = *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)));
  if ( !v8 || *v8 != 31 )
  {
    v6 = (__m128i *)sub_39A81B0((__int64)a1, v8);
    if ( v6 )
      goto LABEL_6;
LABEL_17:
    if ( *v8 == 18 && *(_WORD *)(a2 + 2) == 52 )
    {
      v13 = sub_145CDC0(0x30u, a1 + 11);
      v6 = (__m128i *)v13;
      if ( v13 )
      {
        *(_BYTE *)(v13 + 30) = 0;
        *(_QWORD *)v13 = v13 | 4;
        *(_QWORD *)(v13 + 8) = 0;
        *(_QWORD *)(v13 + 16) = 0;
        *(_DWORD *)(v13 + 24) = -1;
        *(_WORD *)(v13 + 28) = 52;
        *(_QWORD *)(v13 + 32) = 0;
        *(_QWORD *)(v13 + 40) = 0;
      }
      sub_39A55B0((__int64)a1, (unsigned __int8 *)a2, (unsigned __int8 *)v13);
      sub_39A6490((__int64)a1, (__int64)v8, (__int64)v6, v14, v15, v16);
    }
    goto LABEL_7;
  }
  v6 = sub_39CBD10(a1, (__int64)v8, a3, a4);
  if ( !v6 )
    goto LABEL_17;
LABEL_6:
  v6 = (__m128i *)sub_39A5A90((__int64)a1, *(_WORD *)(a2 + 2), (__int64)v6, (unsigned __int8 *)a2);
LABEL_7:
  v9 = *(unsigned int *)(a2 + 8);
  v10 = *(_QWORD *)(a2 + 8 * (6 - v9));
  if ( v10 )
  {
    v23 = *(unsigned __int8 **)(v10 + 8 * (1LL - *(unsigned int *)(v10 + 8)));
    v11 = sub_39A8670(a1, v10);
    sub_39A3B20((__int64)a1, (__int64)v6, 71, (__int64)v11);
    if ( v24 != *(_QWORD *)(v10 + 8 * (3LL - *(unsigned int *)(v10 + 8))) )
      sub_39A6760(a1, (__int64)v6, v24, 73);
  }
  else
  {
    v23 = *(unsigned __int8 **)(a2 - 8 * v9);
    v20 = *(void **)(a2 + 8 * (4 - v9));
    if ( v20 )
    {
      v20 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (4 - v9)));
      v22 = v21;
    }
    else
    {
      v22 = 0;
    }
    sub_39A3F30(a1, (__int64)v6, 3, v20, v22);
    sub_39A6760(a1, (__int64)v6, v24, 73);
    if ( !*(_BYTE *)(a2 + 32) )
      sub_39A34D0((__int64)a1, (__int64)v6, 63);
    sub_39A37D0((__int64)a1, (__int64)v6, a2);
  }
  if ( *(_BYTE *)(a2 + 33) )
  {
    v17 = *(unsigned int *)(a2 + 8);
    v18 = *(_BYTE **)(a2 + 8 * (1 - v17));
    if ( v18 )
      v18 = (_BYTE *)sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v17)));
    else
      v19 = 0;
    sub_39C8580((__int64)a1, v18, v19, (__int64)v6, v23);
  }
  else
  {
    sub_39A34D0((__int64)a1, (__int64)v6, 60);
  }
  if ( (unsigned __int16)sub_398C0A0(a1[25]) > 4u )
  {
    v12 = *(_DWORD *)(a2 + 28) >> 3;
    if ( v12 )
    {
      v26[0] = 65551;
      sub_39A3560((__int64)a1, &v6->m128i_i64[1], 136, (__int64)v26, v12 & 0x1FFFFFFF);
    }
  }
  sub_39CB550(a1, v6, a2, a3, a4);
  return v6;
}
