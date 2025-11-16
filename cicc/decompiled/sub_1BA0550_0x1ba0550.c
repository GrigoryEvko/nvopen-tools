// Function: sub_1BA0550
// Address: 0x1ba0550
//
void __fastcall sub_1BA0550(__int64 a1, __int64 a2, __m128i a3, __m128i a4, double a5)
{
  __int64 v5; // rbx
  __int64 i; // r12
  unsigned __int64 v8; // rsi

  v5 = *(_QWORD *)(a1 + 40);
  for ( i = *(_QWORD *)(a1 + 48); v5 != i; v5 = *(_QWORD *)(v5 + 8) )
  {
    v8 = v5 - 24;
    if ( !v5 )
      v8 = 0;
    sub_1B9F3B0(*(_QWORD *)(a2 + 224), v8, a3, a4, a5);
  }
}
