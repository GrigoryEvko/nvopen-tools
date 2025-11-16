// Function: sub_AF5140
// Address: 0xaf5140
//
__int64 __fastcall sub_AF5140(__int64 a1, unsigned int a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdi

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
    v3 = *(_QWORD *)(a1 - 32);
  else
    v3 = a1 - 16 - 8LL * ((v2 >> 2) & 0xF);
  return *(_QWORD *)(v3 + 8LL * a2);
}
