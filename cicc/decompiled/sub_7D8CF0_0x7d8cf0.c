// Function: sub_7D8CF0
// Address: 0x7d8cf0
//
void __fastcall sub_7D8CF0(__m128i *a1)
{
  __int8 v2; // al
  __int64 i; // rbx

  while ( 1 )
  {
    switch ( a1[10].m128i_i8[13] )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 8:
      case 0xD:
        return;
      case 4:
        sub_7D8B10(a1);
        return;
      case 5:
        a1[10].m128i_i8[13] = 3;
        return;
      case 6:
        v2 = a1[11].m128i_i8[0];
        if ( (unsigned __int8)v2 > 2u && v2 != 6 )
          goto LABEL_7;
        return;
      case 0xA:
        if ( (unsigned int)sub_8D2B50(a1[8].m128i_i64[0]) )
          sub_7D8C20(a1);
        for ( i = a1[11].m128i_i64[0]; i; i = *(_QWORD *)(i + 120) )
          sub_7D8CF0(i);
        return;
      case 0xB:
        a1 = (__m128i *)a1[11].m128i_i64[0];
        break;
      default:
LABEL_7:
        sub_721090();
    }
  }
}
