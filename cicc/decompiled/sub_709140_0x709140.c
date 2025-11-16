// Function: sub_709140
// Address: 0x709140
//
__int64 __fastcall sub_709140(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r12
  __int64 i; // rbx

  v2 = *(_QWORD *)(unk_4D03FF0 + 8LL);
  sub_8252A0();
  sub_7CA8F0();
  if ( unk_4D03FE8 && !dword_4D04944 && qword_4D0495C )
    sub_603790(a1, a2);
  sub_861D10(v2, 0, unk_4D03FF0 + 24LL, 1, 0, 0);
  sub_862250(v2);
  if ( dword_4F077C4 == 2 )
    sub_72D210();
  if ( (unsigned int)sub_7E16F0() )
  {
    sub_8163B0(0);
    if ( unk_4D03B70 )
      sub_708DA0(*(_QWORD *)(unk_4D03FF0 + 8LL));
  }
  if ( dword_4F077C4 == 2 && !dword_4F07590 )
  {
    for ( i = *(_QWORD *)(v2 + 272); i; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_BYTE *)(i + 89) & 4) != 0 && (unsigned int)sub_734480(i) )
      {
        *(_BYTE *)(i + 89) &= ~4u;
        *(_QWORD *)(i + 40) = 0;
      }
    }
  }
  return sub_77F950();
}
