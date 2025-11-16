// Function: sub_ED8700
// Address: 0xed8700
//
__int64 __fastcall sub_ED8700(__int64 a1)
{
  _QWORD *v1; // rdx
  unsigned int v2; // r8d

  v1 = *(_QWORD **)(a1 + 8);
  v2 = 0;
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v1 > 7u )
    LOBYTE(v2) = *v1 == 0x8169666F72706CFFLL;
  return v2;
}
