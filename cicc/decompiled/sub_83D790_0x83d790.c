// Function: sub_83D790
// Address: 0x83d790
//
__int64 __fastcall sub_83D790(__int64 a1, __int64 *a2, __m128i *a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r15
  __m128i *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // rax
  __m128i *v16; // rcx
  int v17; // eax
  __int64 v18; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __m128i *v22; // [rsp-10h] [rbp-90h]
  unsigned int v24; // [rsp+Ch] [rbp-74h]
  __m128i *v25[2]; // [rsp+18h] [rbp-68h] BYREF
  int v26; // [rsp+2Ch] [rbp-54h] BYREF
  const __m128i *v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h] BYREF
  __m128i *v29; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = 0;
  v9 = *(__m128i **)(a1 + 8);
  v25[0] = a3;
  v30[0] = 0;
  v27 = v9;
  if ( a2 )
    v6 = *a2;
  if ( (*(_BYTE *)(a1 + 32) & 0x40) != 0 )
  {
    switch ( *(_BYTE *)(a4 + 80) )
    {
      case 4:
      case 5:
        v11 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 80LL);
        goto LABEL_11;
      case 6:
        v11 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 32LL);
        goto LABEL_11;
      case 9:
      case 0xA:
        v11 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 56LL);
LABEL_11:
        v12 = **(_QWORD **)(v11 + 328);
        if ( (*(_BYTE *)(a1 + 33) & 1) == 0 )
          goto LABEL_12;
        goto LABEL_48;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v12 = **(_QWORD **)(*(_QWORD *)(a4 + 88) + 328LL);
        if ( (*(_BYTE *)(a1 + 33) & 1) != 0 )
        {
LABEL_48:
          sub_869480(*(_QWORD *)(a1 + 80), v12, a5, v30);
          v9 = *(__m128i **)(a1 + 8);
        }
LABEL_12:
        v24 = 1;
        v13 = 0;
        break;
      default:
        BUG();
    }
    while ( 1 )
    {
      v27 = v9;
      if ( v6 )
      {
        v14 = *(_BYTE *)(v6 + 8);
        if ( v14 == 1 )
        {
          if ( !(unsigned int)sub_83CDB0(v6, v9, v12, a5) )
            goto LABEL_37;
          goto LABEL_13;
        }
        if ( v14 == 2 )
          goto LABEL_37;
        v15 = *(_QWORD *)(v6 + 24);
        a3 = (__m128i *)qword_4D03C50;
        v13 = v15 + 8;
        if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
        {
          if ( !*(_BYTE *)(v15 + 24) )
            goto LABEL_37;
          v16 = *(__m128i **)(v15 + 8);
          a3 = (__m128i *)v16[8].m128i_u8[12];
          if ( (_BYTE)a3 == 12 )
          {
            v20 = *(_QWORD *)(v15 + 8);
            do
            {
              v20 = *(_QWORD *)(v20 + 160);
              a3 = (__m128i *)*(unsigned __int8 *)(v20 + 140);
            }
            while ( (_BYTE)a3 == 12 );
          }
          if ( !(_BYTE)a3 )
          {
LABEL_37:
            v24 = 0;
LABEL_28:
            v18 = v30[0];
            if ( v30[0] )
LABEL_29:
              sub_866BE0(v18, v9, a3);
            goto LABEL_30;
          }
        }
        else
        {
          v16 = *(__m128i **)(v15 + 8);
        }
        v25[0] = v16;
      }
      v9 = (__m128i *)v25;
      v17 = sub_82C250(&v27, v25, v13, v12, *a5, &v28, &v29, &v26);
      a3 = v22;
      if ( v17 )
      {
        v9 = v25[0];
        v24 = sub_828690((__int64)v27, (__int64)v25[0]->m128i_i64, v28, (__int64)v29, (__int64)a5, v12);
        if ( !v24 )
          goto LABEL_28;
      }
      else if ( !v26 || (*(_BYTE *)(a1 + 33) & 1) != 0 )
      {
        goto LABEL_37;
      }
      if ( !v6 )
        goto LABEL_28;
LABEL_13:
      v10 = *(_QWORD *)v6;
      if ( !*(_QWORD *)v6 )
      {
        v6 = 0;
        goto LABEL_28;
      }
      if ( *(_BYTE *)(v10 + 8) == 3 )
      {
        v21 = sub_6BBB10((_QWORD *)v6);
        v18 = v30[0];
        v6 = v21;
        if ( !v30[0] )
          goto LABEL_30;
        if ( !v21 )
          goto LABEL_29;
      }
      else
      {
        if ( !v30[0] )
          goto LABEL_9;
        v6 = *(_QWORD *)v6;
      }
      sub_866B90();
      v9 = *(__m128i **)(a1 + 8);
    }
  }
  if ( (*(_BYTE *)(a1 + 33) & 3) == 1 )
  {
    v24 = *(_QWORD *)a1 == 0;
    if ( !*(_QWORD *)a1 )
      v6 = 0;
  }
  else
  {
    v24 = 1;
    if ( v6 )
    {
      v10 = *(_QWORD *)v6;
      if ( *(_QWORD *)v6 )
      {
        if ( *(_BYTE *)(v10 + 8) == 3 )
        {
          v6 = sub_6BBB10((_QWORD *)v6);
          goto LABEL_28;
        }
LABEL_9:
        v6 = v10;
      }
      else
      {
        v6 = 0;
      }
    }
  }
LABEL_30:
  if ( a2 )
    *a2 = v6;
  return v24;
}
