// Function: sub_7F2A70
// Address: 0x7f2a70
//
void __fastcall sub_7F2A70(__m128i *a1, int a2)
{
  __int64 v2; // r13
  __int8 v3; // al
  int v4[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = (__int64)a1;
  v3 = a1[1].m128i_i8[8];
  if ( v3 == 10 )
  {
    v2 = a1[3].m128i_i64[1];
    v3 = *(_BYTE *)(v2 + 24);
  }
  if ( v3 == 1 && *(_BYTE *)(v2 + 56) == 20 && (unsigned int)sub_8D33B0(**(_QWORD **)(v2 + 72)) )
    sub_730620(v2, *(const __m128i **)(v2 + 72));
  if ( dword_4D0439C )
    sub_7DF820((__int64)a1, v4, 0);
  if ( a2 )
  {
    sub_7F2600((__int64)a1, 0);
    sub_7F0F50((__int64)a1, 0);
    if ( unk_4F07520 )
      sub_7E67B0(a1);
  }
  else
  {
    sub_7EE560(a1, 0);
    sub_7F0F50((__int64)a1, 0);
  }
}
