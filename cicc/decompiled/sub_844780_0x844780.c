// Function: sub_844780
// Address: 0x844780
//
__int64 __fastcall sub_844780(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  const __m128i *v7; // rbx
  __int64 v8; // r8
  __int8 v9; // al
  __int8 v10; // al
  char v11; // al
  __int64 v12; // r8
  __int64 v14; // rax
  __int64 v15; // rax
  char i; // dl
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  const __m128i *v23; // rax
  __m128i v24[24]; // [rsp+0h] [rbp-180h] BYREF

  v7 = (const __m128i *)sub_737590(a1, a2, a3, a4, a5);
  v9 = v7[1].m128i_i8[8];
  if ( v9 == 1 )
  {
    while ( 1 )
    {
      v10 = v7[3].m128i_i8[8];
      if ( v10 != 7 )
        break;
      v7 = (const __m128i *)sub_737590((__int64 *)v7[4].m128i_i64[1], a2, v5, v6, v8);
      v9 = v7[1].m128i_i8[8];
      if ( v9 != 1 )
        goto LABEL_11;
    }
    if ( v10 == 4 )
    {
      v14 = v7[4].m128i_i64[1];
      if ( *(_BYTE *)(v14 + 24) != 1 || *(_BYTE *)(v14 + 56) != 1 )
      {
        v9 = v7[1].m128i_i8[8];
        if ( v9 != 1 )
          goto LABEL_11;
        v11 = 4;
LABEL_20:
        if ( (_DWORD)a2 && (v7[1].m128i_i8[9] & 3) == 0 && (unsigned __int8)(v11 - 105) <= 4u )
        {
          v15 = v7->m128i_i64[0];
          for ( i = *(_BYTE *)(v7->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v15 + 140) )
            v15 = *(_QWORD *)(v15 + 160);
          if ( (unsigned __int8)(i - 9) <= 2u )
          {
            v17 = *(_QWORD *)(*(_QWORD *)v15 + 96LL);
            v18 = (__int64 *)sub_730FF0(v7);
            sub_6E70E0(v18, (__int64)v24);
            if ( (*(_BYTE *)(v17 + 177) & 0x40) != 0 )
              sub_8283A0((__int64)v24, v7->m128i_i64[0], 0, 0);
            else
              sub_844770(v24, 0);
            v23 = (const __m128i *)sub_6F6F40(v24, 0, v19, v20, v21, v22);
            sub_730620((__int64)v7, v23);
          }
        }
LABEL_16:
        v9 = v7[1].m128i_i8[8];
        goto LABEL_11;
      }
      v7 = *(const __m128i **)(v14 + 72);
    }
    while ( 2 )
    {
      v9 = v7[1].m128i_i8[8];
      if ( v9 != 1 )
        break;
      while ( 1 )
      {
        v11 = v7[3].m128i_i8[8];
        if ( v11 == 94 )
          goto LABEL_14;
        if ( v11 == 14 )
          break;
        if ( v11 != 91 )
          goto LABEL_20;
        v7 = *(const __m128i **)(v7[4].m128i_i64[1] + 16);
        v9 = v7[1].m128i_i8[8];
        if ( v9 != 1 )
          goto LABEL_11;
      }
      if ( (v7[1].m128i_i8[9] & 3) != 0 )
      {
LABEL_14:
        v7 = (const __m128i *)v7[4].m128i_i64[1];
        continue;
      }
      goto LABEL_16;
    }
  }
LABEL_11:
  v12 = 0;
  if ( (unsigned __int8)(v9 - 5) <= 1u )
    return v7[3].m128i_i64[1];
  return v12;
}
