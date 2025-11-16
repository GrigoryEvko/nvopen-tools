// Function: sub_22AF4B0
// Address: 0x22af4b0
//
__int64 __fastcall sub_22AF4B0(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 80) )
    return *(unsigned int *)(a1 + 76);
  else
    return *(_WORD *)(*(_QWORD *)(a1 + 16) + 2LL) & 0x3F;
}
