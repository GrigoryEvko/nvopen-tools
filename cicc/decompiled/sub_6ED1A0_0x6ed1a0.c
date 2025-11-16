// Function: sub_6ED1A0
// Address: 0x6ed1a0
//
void __fastcall sub_6ED1A0(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 17) == 1 && !sub_6ED0A0(a1) )
    *(_BYTE *)(*(_QWORD *)(a1 + 144) + 25LL) = *(_BYTE *)(*(_QWORD *)(a1 + 144) + 25LL) & 0xFC | 2;
}
