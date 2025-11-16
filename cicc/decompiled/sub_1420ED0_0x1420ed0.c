// Function: sub_1420ED0
// Address: 0x1420ed0
//
__int64 __fastcall sub_1420ED0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdx
  unsigned __int8 v5; // al
  unsigned __int64 v7; // rax
  __m128i v8; // xmm4
  __m128i v9; // xmm5
  __m128i v10; // xmm1
  __m128i v11; // xmm6
  __m128i v12; // xmm7
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  _QWORD *v15; // [rsp+8h] [rbp-A8h]
  _QWORD *v16; // [rsp+8h] [rbp-A8h]
  _QWORD *v17; // [rsp+8h] [rbp-A8h]
  _QWORD *v18; // [rsp+8h] [rbp-A8h]
  __m128i v19; // [rsp+10h] [rbp-A0h] BYREF
  __m128i v20; // [rsp+20h] [rbp-90h] BYREF
  __int64 v21; // [rsp+30h] [rbp-80h]
  char v22[8]; // [rsp+40h] [rbp-70h] BYREF
  __m128i v23; // [rsp+48h] [rbp-68h]
  __m128i v24; // [rsp+58h] [rbp-58h]
  __int64 v25; // [rsp+68h] [rbp-48h]
  __m128i v26; // [rsp+70h] [rbp-40h] BYREF
  __m128i v27; // [rsp+80h] [rbp-30h]
  __int64 v28; // [rsp+90h] [rbp-20h]

  v4 = *(_QWORD *)(a2 + 72);
  v22[0] = 0;
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 <= 0x17u )
  {
LABEL_11:
    switch ( *(_BYTE *)(v4 + 16) )
    {
      case '6':
        v16 = a3;
        sub_141EB40(&v19, (__int64 *)v4);
        goto LABEL_14;
      case '7':
        v18 = a3;
        sub_141EDF0(&v19, v4);
        v4 = *(_QWORD *)(a2 + 72);
        v13 = _mm_loadu_si128(&v19);
        v14 = _mm_loadu_si128(&v20);
        v28 = v21;
        a3 = v18;
        v26 = v13;
        v27 = v14;
        break;
      case ':':
        v17 = a3;
        sub_141F110(&v19, v4);
        v4 = *(_QWORD *)(a2 + 72);
        v11 = _mm_loadu_si128(&v19);
        v12 = _mm_loadu_si128(&v20);
        v28 = v21;
        a3 = v17;
        v26 = v11;
        v27 = v12;
        break;
      case ';':
        v16 = a3;
        sub_141F3C0(&v19, v4);
LABEL_14:
        v10 = _mm_loadu_si128(&v20);
        v4 = *(_QWORD *)(a2 + 72);
        a3 = v16;
        v26 = _mm_loadu_si128(&v19);
        v28 = v21;
        v27 = v10;
        break;
      case 'R':
        v15 = a3;
        sub_141F0A0(&v19, v4);
        v4 = *(_QWORD *)(a2 + 72);
        v8 = _mm_loadu_si128(&v19);
        v9 = _mm_loadu_si128(&v20);
        v28 = v21;
        a3 = v15;
        v26 = v8;
        v27 = v9;
        break;
      default:
        goto LABEL_7;
    }
    goto LABEL_7;
  }
  if ( v5 == 78 )
  {
    if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
LABEL_7:
      v23 = v26;
      v24 = v27;
      v25 = v28;
      goto LABEL_5;
    }
    v22[0] = 1;
    v7 = v4 | 4;
    goto LABEL_9;
  }
  if ( v5 == 29 )
  {
    if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_11;
    v22[0] = 1;
    v7 = v4 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_9:
    v23.m128i_i64[0] = v7;
    goto LABEL_5;
  }
  if ( v5 != 57 )
    goto LABEL_11;
LABEL_5:
  sub_14205E0((bool *)v26.m128i_i8, a1, v4, (__int64)v22, a3);
  return v26.m128i_u8[0];
}
