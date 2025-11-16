// Function: sub_73EA10
// Address: 0x73ea10
//
__int64 __fastcall sub_73EA10(const __m128i **a1, __int64 *a2)
{
  const __m128i **v3; // r12
  const __m128i *v4; // rdx
  __int64 result; // rax
  __int64 v6; // rcx
  bool v7; // cf
  bool v8; // zf
  unsigned __int64 v9; // rax
  const __m128i *v10; // rbx
  __int64 v11; // r14
  char v12; // di
  _QWORD *v13; // rsi
  __int8 v14; // dl
  __m128i *v15; // rax
  __int8 v16; // al
  unsigned __int64 v17; // rax

  v3 = a1;
  v4 = *a1;
  LOBYTE(result) = (*a1)[8].m128i_i8[12];
  if ( (_BYTE)result == 12 && v4[11].m128i_i8[8] == 8 )
  {
    v3 = (const __m128i **)&v4[10];
    v4 = (const __m128i *)v4[10].m128i_i64[0];
    LOBYTE(result) = v4[8].m128i_i8[12];
  }
  v6 = 6338;
  while ( 1 )
  {
    v7 = (unsigned __int8)result < 0xCu;
    v8 = (_BYTE)result == 12;
    if ( (_BYTE)result == 12 )
      break;
    while ( !v7 && !v8 )
    {
      if ( (_BYTE)result != 13 )
        goto LABEL_35;
      v4 = (const __m128i *)v4[10].m128i_i64[1];
      result = v4[8].m128i_u8[12];
      if ( (_BYTE)result == 7 )
        goto LABEL_7;
      v7 = (unsigned __int8)result < 0xCu;
      v8 = (_BYTE)result == 12;
      if ( (_BYTE)result == 12 )
        goto LABEL_11;
    }
    if ( (_BYTE)result == 6 )
    {
LABEL_6:
      v4 = (const __m128i *)v4[10].m128i_i64[0];
      result = v4[8].m128i_u8[12];
      if ( (_BYTE)result == 7 )
      {
LABEL_7:
        *a2 = (__int64)v4;
        return result;
      }
    }
    else
    {
      if ( (_BYTE)result != 7 )
        goto LABEL_35;
      *a2 = (__int64)v4;
      result = v4[8].m128i_u8[12];
      if ( (_BYTE)result == 7 )
      {
        *a2 = (__int64)v4;
        return result;
      }
    }
  }
LABEL_11:
  if ( !v4->m128i_i64[1] )
  {
    v9 = v4[11].m128i_u8[8];
    if ( (unsigned __int8)v9 > 0xCu || !_bittest64(&v6, v9) )
      goto LABEL_6;
  }
  v10 = *v3;
  v11 = 6338;
  while ( 1 )
  {
    while ( 1 )
    {
      v12 = v10[8].m128i_i8[12];
      if ( v12 != 12 )
        goto LABEL_22;
LABEL_16:
      if ( v10->m128i_i64[1] )
      {
        v13 = (_QWORD *)v10[6].m128i_i64[1];
        if ( !v13 )
          goto LABEL_45;
        goto LABEL_18;
      }
      v17 = v10[11].m128i_u8[8];
      if ( (unsigned __int8)v17 <= 0xCu )
      {
        if ( _bittest64(&v11, v17) )
          break;
      }
      while ( 1 )
      {
LABEL_22:
        v15 = (__m128i *)sub_7259C0(v12);
        *v3 = v15;
        sub_73C230(v10, v15);
        v16 = v10[8].m128i_i8[12];
        if ( v16 == 7 )
        {
          result = (__int64)*v3;
          goto LABEL_37;
        }
        if ( (unsigned __int8)v16 > 7u )
          break;
        if ( v16 != 6 )
          goto LABEL_35;
        result = (__int64)*v3;
LABEL_21:
        v10 = (const __m128i *)v10[10].m128i_i64[0];
        v3 = (const __m128i **)(result + 160);
        v12 = v10[8].m128i_i8[12];
        if ( v12 == 12 )
          goto LABEL_16;
      }
      if ( v16 != 12 )
      {
        if ( v16 == 13 )
        {
          result = (__int64)*v3;
          goto LABEL_27;
        }
LABEL_35:
        sub_721090();
      }
LABEL_39:
      v10 = (const __m128i *)v10[10].m128i_i64[0];
      if ( *v3 != v10 )
        v3 = (const __m128i **)&(*v3)[10];
    }
    v13 = (_QWORD *)v10[6].m128i_i64[1];
    if ( !v13 )
      goto LABEL_45;
LABEL_18:
    result = sub_5CEF40(v10[10].m128i_i64[0], v13);
    *v3 = (const __m128i *)result;
    v14 = v10[8].m128i_i8[12];
    if ( v14 == 7 )
      break;
    if ( (unsigned __int8)v14 <= 7u )
    {
      if ( v14 == 6 )
        goto LABEL_21;
      goto LABEL_35;
    }
    if ( v14 == 12 )
    {
      if ( v10[6].m128i_i64[1] )
        goto LABEL_39;
LABEL_45:
      v10 = (const __m128i *)v10[10].m128i_i64[0];
    }
    else
    {
      if ( v14 != 13 )
        goto LABEL_35;
LABEL_27:
      v10 = (const __m128i *)v10[10].m128i_i64[1];
      v3 = (const __m128i **)(result + 168);
    }
  }
LABEL_37:
  *a2 = result;
  return result;
}
