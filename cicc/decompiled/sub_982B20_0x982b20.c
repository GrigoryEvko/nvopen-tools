// Function: sub_982B20
// Address: 0x982b20
//
void __fastcall sub_982B20(__m128i **a1, int a2, __int64 a3)
{
  unsigned int v3; // eax

  switch ( a2 )
  {
    case 1:
      sub_982A00(a1, (const __m128i *)&qword_4F7F280, 38);
      break;
    case 2:
      sub_982A00(a1, (const __m128i *)&qword_4F7E400, 58);
      break;
    case 3:
      sub_982A00(a1, (const __m128i *)&qword_4F7D500, 60);
      break;
    case 4:
      sub_982A00(a1, (const __m128i *)&qword_4F7C580, 62);
      break;
    case 5:
      sub_982A00(a1, (const __m128i *)&qword_4F79E80, 156);
      break;
    case 6:
      v3 = *(_DWORD *)(a3 + 32);
      if ( v3 > 4 )
      {
        if ( v3 == 29 )
          sub_982A00(a1, (const __m128i *)&qword_4F74380, 124);
      }
      else if ( v3 > 2 )
      {
        sub_982A00(a1, (const __m128i *)&qword_4F78F80, 60);
        sub_982A00(a1, (const __m128i *)&qword_4F78080, 60);
        sub_982A00(a1, (const __m128i *)&qword_4F76280, 120);
      }
      break;
    case 7:
      if ( (unsigned int)(*(_DWORD *)(a3 + 32) - 3) <= 1 )
        sub_982A00(a1, (const __m128i *)&qword_4F70780, 240);
      break;
    case 8:
      sub_982A00(a1, (const __m128i *)&qword_4F6D400, 206);
      break;
    default:
      return;
  }
}
