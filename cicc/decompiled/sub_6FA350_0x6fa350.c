// Function: sub_6FA350
// Address: 0x6fa350
//
void __fastcall sub_6FA350(const __m128i *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9

  if ( a1[1].m128i_i8[1] == 2 )
  {
    if ( (unsigned int)sub_8D3A70(a1->m128i_i64[0]) )
      sub_6FA340((__int64)a1, a2, v2, v3, v4, v5);
    else
      sub_6EFA90(a1, a2);
  }
}
