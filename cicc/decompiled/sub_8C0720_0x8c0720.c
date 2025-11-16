// Function: sub_8C0720
// Address: 0x8c0720
//
void __fastcall sub_8C0720(__int64 a1, __int64 a2, _QWORD *a3, __m128i *a4, __int64 a5, int *a6, __m128i *a7)
{
  __int64 v7; // rbx
  char v11; // al
  unsigned __int8 v12; // di
  _QWORD *v13; // r15
  char v14; // al
  bool v15; // zf
  __int64 v16; // r15
  __int64 v17; // r10
  _QWORD *v18; // rax
  char v19; // al
  __int64 **v20; // rax
  const __m128i *v21; // [rsp+0h] [rbp-50h]
  __m128i *v22; // [rsp+8h] [rbp-48h]
  __int64 *v24; // [rsp+18h] [rbp-38h]

  v24 = (__int64 *)(a1 + 48);
  if ( a2 )
  {
    v7 = a2;
    do
    {
      while ( 1 )
      {
        v11 = *(_BYTE *)(*(_QWORD *)(v7 + 8) + 80LL);
        if ( v11 == 3 )
          break;
        if ( v11 != 2 )
        {
          if ( (*(_BYTE *)(v7 + 56) & 1) != 0 )
            goto LABEL_6;
          goto LABEL_15;
        }
        if ( (*(_BYTE *)(v7 + 72) & 1) != 0 )
        {
          v20 = sub_8A2270(*(_QWORD *)(*(_QWORD *)(v7 + 64) + 128LL), a4, (__int64)a3, v24, 24576, a6, a7);
          if ( *a6 )
            return;
          *(_QWORD *)(*(_QWORD *)(v7 + 64) + 128LL) = v20;
LABEL_20:
          if ( (*(_BYTE *)(v7 + 56) & 1) != 0 )
          {
            v19 = *(_BYTE *)(*(_QWORD *)(v7 + 8) + 80LL);
            if ( v19 == 3 )
            {
              v12 = 0;
            }
            else if ( v19 == 2 )
            {
LABEL_24:
              v12 = 1;
            }
            else
            {
LABEL_6:
              v12 = 2;
            }
LABEL_7:
            v13 = sub_725090(v12);
            sub_8AEEA0(a1, (__int64)v13, v7, a3);
            sub_8A4F30((__int64)v13, v7, a4, (__int64)a3, a4, (__int64)a3, v24, 0x6000u, a6, a7);
            if ( *a6 )
              return;
            v14 = *((_BYTE *)v13 + 8);
            if ( v14 != 1 && v14 != 2 && v14 )
              sub_721090();
            v15 = *(_QWORD *)(v7 + 88) == 0;
            *(_QWORD *)(v7 + 80) = v13[4];
            if ( !v15 )
              sub_88EF90(v7);
            sub_725130(v13);
            goto LABEL_15;
          }
          goto LABEL_15;
        }
        if ( (*(_BYTE *)(v7 + 56) & 1) != 0 )
          goto LABEL_24;
        v7 = *(_QWORD *)v7;
        if ( !v7 )
          return;
      }
      v16 = *(_QWORD *)(v7 + 64);
      v17 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 32LL);
      if ( v17 )
      {
        v21 = *(const __m128i **)(*(_QWORD *)(v16 + 168) + 32LL);
        v22 = sub_8A55D0(
                0,
                *(__m128i **)(v17 + 64),
                0,
                0,
                (__int64)a4,
                (__int64)a3,
                (__int64 *)(v17 + 28),
                24576,
                a6,
                a7);
        if ( *a6 )
          return;
        v18 = sub_730FF0(v21);
        v18[8] = v22;
        *(_QWORD *)(*(_QWORD *)(v16 + 168) + 32LL) = v18;
        goto LABEL_20;
      }
      if ( (*(_BYTE *)(v7 + 56) & 1) != 0 )
      {
        v12 = 0;
        goto LABEL_7;
      }
LABEL_15:
      v7 = *(_QWORD *)v7;
    }
    while ( v7 );
  }
}
