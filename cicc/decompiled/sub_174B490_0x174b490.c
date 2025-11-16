// Function: sub_174B490
// Address: 0x174b490
//
__int64 __fastcall sub_174B490(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r13
  char v12; // al
  int v13; // eax
  __int64 v14; // r15
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  unsigned __int8 v21; // al
  char v22; // al
  __int64 *v23; // rbx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 i; // r15
  _QWORD *v27; // rbx
  unsigned __int8 v28; // al
  __int64 *v29; // rbx
  unsigned __int64 v30; // rax
  _QWORD *v31; // rax
  int v32; // [rsp+Ch] [rbp-54h]
  _BYTE v33[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v34; // [rsp+20h] [rbp-40h]

  v11 = *(_QWORD *)(a2 - 24);
  v12 = *(_BYTE *)(v11 + 16);
  if ( (unsigned __int8)(v12 - 60) <= 0xCu )
  {
    v13 = sub_174B310((__int64)a1, *(_QWORD *)(a2 - 24), a2);
    if ( v13 )
    {
      v16 = *(_QWORD *)a2;
      v34 = 257;
      v14 = sub_15FDBD0(v13, *(_QWORD *)(v11 - 24), v16, (__int64)v33, 0);
      v17 = *(_QWORD *)(v11 + 8);
      if ( !v17 || *(_QWORD *)(v17 + 8) )
        return v14;
      goto LABEL_10;
    }
    v12 = *(_BYTE *)(v11 + 16);
  }
  if ( v12 != 79 )
    goto LABEL_5;
  v18 = *(_QWORD *)(v11 - 72);
  if ( (unsigned __int8)(*(_BYTE *)(v18 + 16) - 75) <= 1u && *(_QWORD *)v11 == **(_QWORD **)(v18 - 48) )
    return 0;
  v14 = sub_1707470((__int64)a1, (_BYTE *)a2, v11, *(double *)a3.m128_u64, a4, a5);
  if ( v14 )
  {
LABEL_10:
    sub_1AEBB60(v11, v14, a2, a1[332]);
    return v14;
  }
  v12 = *(_BYTE *)(v11 + 16);
LABEL_5:
  if ( v12 != 77 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) == 61 && sub_1642F90(*(_QWORD *)v11, 1) && (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) == 2 )
  {
    v20 = (_QWORD *)(v11 - 48);
    if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
      v20 = *(_QWORD **)(v11 - 8);
    v21 = *(_BYTE *)(*v20 + 16LL);
    if ( v21 <= 0x17u )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v20[3] + 16LL) - 75) > 1u )
        goto LABEL_16;
      if ( v21 != 13 )
        goto LABEL_16;
      v29 = &v20[3 * *(unsigned int *)(v11 + 56) + 1];
      v30 = sub_157EBA0(*v29);
      if ( *(_BYTE *)(v30 + 16) != 26 )
        goto LABEL_16;
      if ( (*(_DWORD *)(v30 + 20) & 0xFFFFFFF) == 1 )
        goto LABEL_16;
      if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v30 - 72) + 16LL) - 75) > 1u )
        goto LABEL_16;
      v25 = sub_157EBA0(v29[1]);
      if ( *(_BYTE *)(v25 + 16) != 26 )
        goto LABEL_16;
    }
    else
    {
      if ( (unsigned __int8)(v21 - 75) > 1u )
        goto LABEL_16;
      v22 = *(_BYTE *)(v20[3] + 16LL);
      if ( (unsigned __int8)(v22 - 75) <= 1u )
        goto LABEL_16;
      if ( v22 != 13 )
        goto LABEL_16;
      v23 = &v20[3 * *(unsigned int *)(v11 + 56) + 1];
      v24 = sub_157EBA0(v20[3 * *(unsigned int *)(v11 + 56) + 2]);
      if ( *(_BYTE *)(v24 + 16) != 26 )
        goto LABEL_16;
      if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) == 1 )
        goto LABEL_16;
      if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v24 - 72) + 16LL) - 75) > 1u )
        goto LABEL_16;
      v25 = sub_157EBA0(*v23);
      if ( *(_BYTE *)(v25 + 16) != 26 )
        goto LABEL_16;
    }
    if ( (*(_DWORD *)(v25 + 20) & 0xFFFFFFF) == 1 )
    {
      v32 = 0;
      for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
      {
        v27 = sub_1648700(i);
        v28 = *((_BYTE *)v27 + 16);
        if ( v28 <= 0x17u )
          goto LABEL_16;
        if ( (unsigned __int8)(v28 - 50) <= 1u )
          return 0;
        if ( v28 == 62 && (unsigned int)sub_1648EF0((__int64)v27) == 1 )
        {
          v31 = sub_1648700(v27[1]);
          if ( !v31 )
            goto LABEL_16;
          if ( (unsigned __int8)(*((_BYTE *)v31 + 16) - 50) <= 1u )
            return 0;
        }
        if ( *((_BYTE *)v27 + 16) == 75 )
        {
          ++v32;
          if ( (unsigned int)sub_1648EF0((__int64)v27) == 1 && *((_BYTE *)sub_1648700(v27[1]) + 16) == 77 )
            return 0;
        }
      }
      if ( v32 > 1 )
        return 0;
    }
  }
LABEL_16:
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 11 )
  {
    v19 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 11 && !(unsigned __int8)sub_1705440((__int64)a1, v19, *(_QWORD *)v11) )
      return 0;
  }
  return sub_17127D0(a1, a2, v11, a3, a4, a5, a6, a7, a8, a9, a10);
}
