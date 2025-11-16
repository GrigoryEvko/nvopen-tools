// Function: sub_7032B0
// Address: 0x7032b0
//
__int64 __fastcall sub_7032B0(__int64 a1, __m128i *a2, __m128i *a3, __int64 *a4, unsigned int a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // r9
  __int8 v17; // al
  __int64 result; // rax
  __int8 v19; // al
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  __int8 v23; // al
  __int64 v24; // rcx
  __int64 v25; // rax
  char i; // dl
  __int8 v27; // al
  __int8 v28; // al
  __int64 v30; // [rsp+8h] [rbp-38h]

  v30 = dword_4D03B80;
  if ( !(_BYTE)a1 )
  {
    sub_6F8020(a2);
    if ( a2[1].m128i_i8[1] == 2 )
    {
      v24 = a2[1].m128i_u8[0];
      if ( (_BYTE)v24 != 2 || a2[19].m128i_i8[13] != 12 )
      {
        if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
          goto LABEL_3;
        if ( !(_BYTE)v24 )
        {
LABEL_32:
          if ( !(unsigned int)sub_730FB0(0) )
            goto LABEL_6;
          goto LABEL_18;
        }
        goto LABEL_26;
      }
      sub_6F7FE0((__int64)a2, 0, v20, v24, v21, v22);
    }
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
      goto LABEL_3;
    v23 = a2[1].m128i_i8[1];
    if ( v23 == 1 )
    {
      if ( !sub_6ED0A0((__int64)a2) )
      {
        LOBYTE(v24) = a2[1].m128i_i8[0];
        goto LABEL_29;
      }
      v23 = a2[1].m128i_i8[1];
    }
    LOBYTE(v24) = a2[1].m128i_i8[0];
    if ( v23 != 3 )
    {
      if ( !(_BYTE)v24 )
        goto LABEL_32;
LABEL_26:
      v25 = a2->m128i_i64[0];
      for ( i = *(_BYTE *)(a2->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v25 + 140) )
        v25 = *(_QWORD *)(v25 + 160);
      if ( i )
      {
        sub_6E68E0(0x9Eu, (__int64)a2);
        goto LABEL_32;
      }
    }
LABEL_29:
    if ( (_BYTE)v24 == 3 )
    {
      sub_6F3BA0(a2, 0);
    }
    else if ( (_BYTE)v24 == 4 )
    {
      sub_6EE880((__int64)a2, 0);
    }
    goto LABEL_32;
  }
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
  {
    v19 = a2[1].m128i_i8[0];
    if ( v19 == 3 )
    {
      sub_6F3BA0(a2, 1);
    }
    else
    {
      if ( v19 == 4 )
        sub_6EE880((__int64)a2, 0);
      sub_6F69D0(a2, 0);
    }
    a1 = (unsigned __int8)a1;
    if ( !(unsigned int)sub_730FB0((unsigned __int8)a1) )
      goto LABEL_20;
    goto LABEL_18;
  }
LABEL_3:
  if ( !(unsigned int)sub_68FE10(a2, 1, 0) )
  {
    v11 = sub_736D90((unsigned __int8)a1);
    sub_6F3DD0((__int64)a2, v11, v11 == 0, v12, v13, v14);
    if ( !(unsigned int)sub_730FB0((unsigned __int8)a1) )
      goto LABEL_5;
LABEL_18:
    v30 = sub_6EFF80();
    if ( (_BYTE)a1 )
      goto LABEL_19;
    goto LABEL_6;
  }
  sub_6F40C0((__int64)a2, 1, v7, v8, v9, v10);
LABEL_5:
  if ( (_BYTE)a1 )
  {
LABEL_19:
    a1 = (unsigned __int8)a1;
LABEL_20:
    sub_6FEAC0(a1, a2, v30, a3, a4, a5);
    goto LABEL_12;
  }
LABEL_6:
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
    goto LABEL_19;
  v17 = a2[1].m128i_i8[1];
  if ( v17 == 1 )
  {
    if ( !sub_6ED0A0((__int64)a2) )
    {
      *a3 = _mm_loadu_si128(a2);
      a3[1] = _mm_loadu_si128(a2 + 1);
      a3[2] = _mm_loadu_si128(a2 + 2);
      a3[3] = _mm_loadu_si128(a2 + 3);
      a3[4] = _mm_loadu_si128(a2 + 4);
      a3[5] = _mm_loadu_si128(a2 + 5);
      a3[6] = _mm_loadu_si128(a2 + 6);
      a3[7] = _mm_loadu_si128(a2 + 7);
      a3[8] = _mm_loadu_si128(a2 + 8);
      v28 = a2[1].m128i_i8[0];
      if ( v28 == 2 )
      {
        a3[9] = _mm_loadu_si128(a2 + 9);
        a3[10] = _mm_loadu_si128(a2 + 10);
        a3[11] = _mm_loadu_si128(a2 + 11);
        a3[12] = _mm_loadu_si128(a2 + 12);
        a3[13] = _mm_loadu_si128(a2 + 13);
        a3[14] = _mm_loadu_si128(a2 + 14);
        a3[15] = _mm_loadu_si128(a2 + 15);
        a3[16] = _mm_loadu_si128(a2 + 16);
        a3[17] = _mm_loadu_si128(a2 + 17);
        a3[18] = _mm_loadu_si128(a2 + 18);
        a3[19] = _mm_loadu_si128(a2 + 19);
        a3[20] = _mm_loadu_si128(a2 + 20);
        a3[21] = _mm_loadu_si128(a2 + 21);
      }
      else if ( v28 == 5 || v28 == 1 )
      {
        a3[9].m128i_i64[0] = a2[9].m128i_i64[0];
      }
      sub_6FF5E0((__int64)a3, a4);
      goto LABEL_10;
    }
    v17 = a2[1].m128i_i8[1];
  }
  if ( v17 == 3 )
  {
    *a3 = _mm_loadu_si128(a2);
    a3[1] = _mm_loadu_si128(a2 + 1);
    a3[2] = _mm_loadu_si128(a2 + 2);
    a3[3] = _mm_loadu_si128(a2 + 3);
    a3[4] = _mm_loadu_si128(a2 + 4);
    a3[5] = _mm_loadu_si128(a2 + 5);
    a3[6] = _mm_loadu_si128(a2 + 6);
    a3[7] = _mm_loadu_si128(a2 + 7);
    a3[8] = _mm_loadu_si128(a2 + 8);
    v27 = a2[1].m128i_i8[0];
    if ( v27 == 2 )
    {
      a3[9] = _mm_loadu_si128(a2 + 9);
      a3[10] = _mm_loadu_si128(a2 + 10);
      a3[11] = _mm_loadu_si128(a2 + 11);
      a3[12] = _mm_loadu_si128(a2 + 12);
      a3[13] = _mm_loadu_si128(a2 + 13);
      a3[14] = _mm_loadu_si128(a2 + 14);
      a3[15] = _mm_loadu_si128(a2 + 15);
      a3[16] = _mm_loadu_si128(a2 + 16);
      a3[17] = _mm_loadu_si128(a2 + 17);
      a3[18] = _mm_loadu_si128(a2 + 18);
      a3[19] = _mm_loadu_si128(a2 + 19);
      a3[20] = _mm_loadu_si128(a2 + 20);
      a3[21] = _mm_loadu_si128(a2 + 21);
    }
    else if ( v27 == 5 || v27 == 1 )
    {
      a3[9].m128i_i64[0] = a2[9].m128i_i64[0];
    }
    sub_6F5FA0(a3, a4, 0, 0, v15, v16);
  }
  else
  {
    sub_6E6260(a3);
  }
LABEL_10:
  if ( (a3[1].m128i_i8[3] & 2) == 0 )
    sub_6E3BA0((__int64)a3, a4, a5, 0);
LABEL_12:
  result = a2[4].m128i_u8[0];
  a3[4].m128i_i8[0] = result;
  return result;
}
