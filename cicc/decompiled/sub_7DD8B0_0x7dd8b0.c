// Function: sub_7DD8B0
// Address: 0x7dd8b0
//
void __fastcall sub_7DD8B0(__m128i *a1, _QWORD *a2)
{
  __m128i *v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rdi
  _QWORD v7[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( a1 )
  {
    v2 = a1;
    do
    {
      if ( !(unsigned int)sub_736DD0((__int64)v2) )
      {
        v3 = (__int64)v2;
        if ( (v2[8].m128i_i8[13] & 0x40) != 0 )
        {
          a2 = v7;
          v3 = sub_7DB130(v2, v7, 0);
          sub_7DC650(v3);
        }
        v4 = *(_QWORD *)(v3 + 152);
        if ( v4 && (unsigned __int8)(*(_BYTE *)(v3 + 140) - 9) <= 2u && *(_BYTE *)(v4 + 136) == 1 )
          sub_7DD730(v3, (__int64)a2);
        if ( (unsigned __int8)(v2[8].m128i_i8[12] - 9) <= 2u )
        {
          v5 = v2[10].m128i_i64[1];
          v6 = *(_QWORD *)(v5 + 152);
          if ( v6 )
          {
            if ( (*(_BYTE *)(v6 + 29) & 0x20) == 0 )
              sub_7DD9B0();
          }
          sub_7DD8B0(*(_QWORD *)(v5 + 216));
        }
      }
      v2 = (__m128i *)v2[7].m128i_i64[0];
    }
    while ( v2 );
  }
}
