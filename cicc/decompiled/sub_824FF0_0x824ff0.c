// Function: sub_824FF0
// Address: 0x824ff0
//
int __fastcall sub_824FF0(__m128i *a1, __int64 a2)
{
  _QWORD *v2; // rax

  LODWORD(v2) = sub_824390(a1);
  if ( !(_DWORD)v2 )
  {
    LODWORD(v2) = sub_8247C0(a2, unk_4D03FA8, (__m128i *)a1[1].m128i_i32);
    if ( !(_DWORD)v2 )
    {
      if ( (unsigned int)sub_824860(a1[2].m128i_i64[0], 3u) )
        sub_824E10((__int64)a1);
      v2 = &qword_4F07280;
      a1->m128i_i64[0] = qword_4F07320[0];
      qword_4F07320[0] = a1;
    }
  }
  return (int)v2;
}
