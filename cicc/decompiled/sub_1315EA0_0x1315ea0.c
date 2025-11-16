// Function: sub_1315EA0
// Address: 0x1315ea0
//
__int64 __fastcall sub_1315EA0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        _QWORD *a5,
        unsigned __int8 a6,
        _QWORD *a7)
{
  unsigned __int8 v7; // r10
  _QWORD *v8; // r11
  _QWORD *v13; // rbx
  unsigned __int64 v14; // rcx
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rsi
  _QWORD *v17; // r12
  unsigned __int64 *v18; // r12
  _QWORD *v19; // r9
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rbx
  bool v22; // si
  unsigned __int64 v23; // r11
  unsigned __int64 v24; // rax
  bool v25; // sf
  int v27; // esi
  int v28; // eax
  __int64 v29; // r8
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rcx
  int v33; // eax
  unsigned __int64 v34; // rdx
  char v35; // cl
  __int64 v36; // rax
  _QWORD *v37; // rdx
  unsigned int i; // edi
  unsigned __int64 v39; // rax
  _QWORD *v41; // [rsp+8h] [rbp-1B8h]
  unsigned __int8 v42; // [rsp+8h] [rbp-1B8h]
  _QWORD v43[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v7 = a6;
  v8 = (_QWORD *)(a1 + 432);
  v13 = a5;
  if ( !a1 )
  {
    sub_130D500(v43);
    v8 = v43;
    v7 = a6;
  }
  v14 = a2 & 0xFFFFFFFFC0000000LL;
  v15 = (_QWORD *)((char *)v8 + ((a2 >> 26) & 0xF0));
  v16 = *v15;
  if ( (a2 & 0xFFFFFFFFC0000000LL) == *v15 )
  {
    v17 = (_QWORD *)(v15[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v14 == v8[32] )
  {
    v30 = v8[33];
LABEL_32:
    v8[32] = v16;
    v8[33] = v15[1];
    v17 = (_QWORD *)(v30 + ((a2 >> 9) & 0x1FFFF8));
    *v15 = v14;
    v15[1] = v30;
  }
  else
  {
    v37 = v8 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v14 == *v37 )
      {
        a5 = &v8[2 * i];
        v8 += 2 * i - 2;
        v30 = a5[33];
        a5[32] = v8[32];
        a5[33] = v8[33];
        goto LABEL_32;
      }
      v37 += 2;
    }
    v42 = v7;
    v39 = sub_130D370(a1, (__int64)&unk_5060AE0, v8, a2, 1, 0);
    v7 = v42;
    v17 = (_QWORD *)v39;
  }
  v18 = (unsigned __int64 *)(((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
  if ( a4 > 0x7000000000000000LL )
  {
    v19 = qword_505FA40;
    v24 = *v18;
    LODWORD(a5) = 1;
    goto LABEL_17;
  }
  if ( a4 > 0x1000 )
  {
    v19 = qword_505FA40;
    _BitScanReverse64((unsigned __int64 *)&v31, 2 * a4 - 1);
    if ( (unsigned __int64)(int)v31 < 7 )
      LOBYTE(v31) = 7;
    v20 = -(1LL << ((unsigned __int8)v31 - 3)) & ((1LL << ((unsigned __int8)v31 - 3)) + a4 - 1);
  }
  else
  {
    v19 = qword_505FA40;
    v20 = qword_505FA40[byte_5060800[(a4 + 7) >> 3]];
  }
  v21 = (unsigned __int64)v13 + a4;
  v22 = a3 <= 0x3800 && v20 <= 0x3800;
  if ( v21 > 0x1000 )
  {
    if ( v21 > 0x7000000000000000LL )
    {
      if ( v22 )
      {
        v23 = 0;
        v27 = byte_5060800[0];
        if ( a3 <= 0x1000 )
          goto LABEL_20;
        goto LABEL_45;
      }
LABEL_25:
      LODWORD(a5) = 1;
      goto LABEL_26;
    }
    _BitScanReverse64((unsigned __int64 *)&v32, 2 * v21 - 1);
    if ( (unsigned __int64)(int)v32 < 7 )
      LOBYTE(v32) = 7;
    v23 = -(1LL << ((unsigned __int8)v32 - 3)) & (v21 + (1LL << ((unsigned __int8)v32 - 3)) - 1);
  }
  else
  {
    v23 = qword_505FA40[byte_5060800[(v21 + 7) >> 3]];
  }
  if ( !v22 )
  {
    if ( a3 > 0x3FFF && v23 > 0x3FFF )
    {
      v33 = sub_1309DD0(a1, v18, v20, v23, v7);
      v19 = qword_505FA40;
      LODWORD(a5) = v33;
      v24 = *v18;
      goto LABEL_17;
    }
    goto LABEL_25;
  }
  if ( v23 > 0x3800 )
    goto LABEL_12;
  if ( v23 > 0x1000 )
  {
    _BitScanReverse64(&v34, 2 * v23 - 1);
    v27 = ((((v23 - 1) & (-1LL << ((unsigned __int8)v34 - 3))) >> ((unsigned __int8)v34 - 3)) & 3) + 4 * v34 - 23;
    if ( a3 <= 0x1000 )
      goto LABEL_20;
  }
  else
  {
    v27 = byte_5060800[(v23 + 7) >> 3];
    if ( a3 <= 0x1000 )
    {
LABEL_20:
      v28 = byte_5060800[(a3 + 7) >> 3];
      goto LABEL_21;
    }
  }
LABEL_45:
  v35 = 7;
  _BitScanReverse64((unsigned __int64 *)&v36, 2 * a3 - 1);
  if ( (unsigned int)v36 >= 7 )
    v35 = v36;
  if ( (unsigned int)v36 < 6 )
    LODWORD(v36) = 6;
  v28 = ((((a3 - 1) & (-1LL << (v35 - 3))) >> (v35 - 3)) & 3) + 4 * v36 - 23;
LABEL_21:
  if ( v28 == v27 )
    goto LABEL_13;
LABEL_12:
  LOBYTE(a5) = a3 > v23 || a4 > a3;
  if ( (_BYTE)a5 )
  {
LABEL_26:
    v24 = *v18;
    goto LABEL_17;
  }
LABEL_13:
  v24 = *v18;
  if ( a1 )
  {
    v25 = --*(_DWORD *)(a1 + 152) < 0;
    if ( v25 && (unsigned __int8)sub_1314130((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
    {
      v41 = v19;
      sub_1315160(a1, v29, 0, 0);
      v24 = *v18;
      v19 = v41;
      LODWORD(a5) = 0;
      goto LABEL_17;
    }
    v24 = *v18;
  }
  LODWORD(a5) = 0;
LABEL_17:
  *a7 = v19[(unsigned __int8)(v24 >> 20)];
  return (unsigned int)a5;
}
