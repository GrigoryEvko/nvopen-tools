// Function: sub_8D5290
// Address: 0x8d5290
//
const __m128i *__fastcall sub_8D5290(__m128i *a1, char a2, _DWORD *a3, int a4)
{
  __m128i *v4; // r14
  const __m128i *result; // rax
  int v8; // esi
  __int64 v9; // rax
  int v10; // r12d
  __int8 v11; // r13
  _QWORD *v12; // rdi
  __int64 v13; // rdi
  const __m128i *v14; // rax

  v4 = a1;
  sub_72C930();
  switch ( a2 )
  {
    case 5:
      if ( !(unsigned int)sub_8D2870((__int64)a1) )
      {
        if ( a4 )
          sub_6851C0(0x8A9u, a3);
        return (const __m128i *)sub_72C930();
      }
      while ( v4[8].m128i_i8[12] == 12 )
        v4 = (__m128i *)v4[10].m128i_i64[0];
      v9 = v4[11].m128i_i64[0];
      if ( (v4[10].m128i_i8[1] & 4) != 0 )
        return *(const __m128i **)(v9 + 8);
      v13 = v4[10].m128i_u8[0];
      if ( (*(_BYTE *)v9 & 4) != 0 )
        LOBYTE(v13) = byte_4B6DF80[v13];
      return (const __m128i *)sub_72BA30(v13);
    case 13:
      if ( !sub_8D3FE0((__int64)a1) )
        return a1;
      if ( (unsigned int)sub_8D2FB0((__int64)a1) )
      {
        v14 = (const __m128i *)sub_8D46C0((__int64)a1);
        v4 = sub_73D660(v14, 3);
      }
      return (const __m128i *)sub_72D600(v4);
    case 14:
      if ( (unsigned int)sub_8D2FB0((__int64)a1) )
        v4 = (__m128i *)sub_8D46C0((__int64)a1);
      return (const __m128i *)sub_72D2E0(v4);
    case 15:
      if ( !sub_8D3FE0((__int64)a1) || (unsigned int)sub_8D2FB0((__int64)a1) )
        return a1;
      return (const __m128i *)sub_72D6A0(a1);
    case 16:
      if ( (unsigned int)sub_8D2FB0((__int64)a1) )
        v4 = (__m128i *)sub_8D46C0((__int64)a1);
      if ( sub_8D3410((__int64)v4) )
      {
        v12 = (_QWORD *)sub_8D4050((__int64)v4);
        return (const __m128i *)sub_72D2E0(v12);
      }
      else if ( sub_8D2310((__int64)v4) )
      {
        return (const __m128i *)sub_72D2E0(v4);
      }
      else
      {
        v8 = 7;
        return sub_73D660(v4, v8);
      }
    case 17:
      if ( !(unsigned int)sub_8D2780((__int64)a1) && !(unsigned int)sub_8D2870((__int64)a1)
        || (unsigned int)sub_8D29A0((__int64)a1) )
      {
        if ( a4 && (unsigned int)sub_6E5430() )
          sub_6851C0(0xCF2u, a3);
        return (const __m128i *)sub_72C930();
      }
      v10 = 0;
      if ( (a1[8].m128i_i8[12] & 0xFB) == 8 )
      {
        v10 = sub_8D4C10((__int64)a1, dword_4F077C4 != 2);
        if ( a1[8].m128i_i8[12] == 12 )
        {
          do
            v4 = (__m128i *)v4[10].m128i_i64[0];
          while ( v4[8].m128i_i8[12] == 12 );
        }
      }
      v11 = v4[10].m128i_i8[0];
      result = (const __m128i *)sub_72BA30(v11);
      if ( !v11 )
      {
        result = (const __m128i *)sub_72BA30(1u);
        goto LABEL_48;
      }
      if ( byte_4B6DF90[v4[10].m128i_u8[0]] )
        goto LABEL_48;
      goto LABEL_42;
    case 18:
      if ( ((unsigned int)sub_8D2780((__int64)a1) || (unsigned int)sub_8D2870((__int64)a1))
        && !(unsigned int)sub_8D29A0((__int64)a1) )
      {
        v10 = 0;
        if ( (a1[8].m128i_i8[12] & 0xFB) == 8 )
        {
          v10 = sub_8D4C10((__int64)a1, dword_4F077C4 != 2);
          if ( a1[8].m128i_i8[12] == 12 )
          {
            do
              v4 = (__m128i *)v4[10].m128i_i64[0];
            while ( v4[8].m128i_i8[12] == 12 );
          }
        }
        v11 = v4[10].m128i_i8[0];
        result = (const __m128i *)sub_72BA30(v11);
        if ( v11 )
        {
          if ( byte_4B6DF90[v4[10].m128i_u8[0]] )
LABEL_42:
            result = (const __m128i *)sub_72BE70(v11);
        }
        else
        {
          result = (const __m128i *)sub_72BA30(2u);
        }
LABEL_48:
        if ( v10 )
          return sub_73C570(result, v10);
      }
      else
      {
        if ( a4 && (unsigned int)sub_6E5430() )
          sub_6851C0(0xCF3u, a3);
        return (const __m128i *)sub_72C930();
      }
      return result;
    case 19:
      if ( !sub_8D3410((__int64)a1) )
        return a1;
      return (const __m128i *)sub_8D40F0((__int64)a1);
    case 20:
      v8 = 1;
      return sub_73D660(v4, v8);
    case 21:
      v8 = 3;
      return sub_73D660(v4, v8);
    case 22:
      return sub_73D6E0(a1);
    case 23:
      if ( !sub_8D3410((__int64)a1) )
        return a1;
      return (const __m128i *)sub_8D4050((__int64)a1);
    case 24:
      if ( (unsigned int)sub_8D2E30((__int64)a1) )
        return (const __m128i *)sub_8D46C0((__int64)a1);
      return a1;
    case 25:
    case 28:
      if ( (unsigned int)sub_8D2FB0((__int64)a1) )
        return (const __m128i *)sub_8D46C0((__int64)a1);
      else
        return a1;
    case 26:
      v8 = 4;
      return sub_73D660(v4, v8);
    case 27:
      v8 = 2;
      return sub_73D660(v4, v8);
    default:
      sub_721090();
  }
}
