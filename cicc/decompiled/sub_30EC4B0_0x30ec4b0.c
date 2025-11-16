// Function: sub_30EC4B0
// Address: 0x30ec4b0
//
void __fastcall sub_30EC4B0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  _BYTE *v3; // rsi

  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v3 = *(_BYTE **)(i + 24);
    if ( *v3 > 0x1Cu )
      sub_30EC400(a1, (__int64)v3);
  }
}
