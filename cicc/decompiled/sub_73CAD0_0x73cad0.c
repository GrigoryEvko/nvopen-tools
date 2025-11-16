// Function: sub_73CAD0
// Address: 0x73cad0
//
__m128i *__fastcall sub_73CAD0(__int64 a1, __int64 a2)
{
  char v2; // dl
  __int64 v3; // rax
  const __m128i *v5; // r13

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 == 12 )
  {
    v3 = a2;
    do
    {
      v3 = *(_QWORD *)(v3 + 160);
      v2 = *(_BYTE *)(v3 + 140);
    }
    while ( v2 == 12 );
  }
  if ( !v2 )
    return (__m128i *)sub_72C930();
  v5 = (const __m128i *)sub_7E1F00(a2);
  if ( (unsigned int)sub_8D2310(v5) )
    return (__m128i *)v5;
  else
    return sub_73CA70(v5, a1);
}
