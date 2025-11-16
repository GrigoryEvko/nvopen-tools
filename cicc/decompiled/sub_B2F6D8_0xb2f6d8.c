// Function: sub_B2F6D8
// Address: 0xb2f6d8
//
__int64 __fastcall sub_B2F6D8(__int64 a1)
{
  if ( *(_QWORD *)(a1 + 40) && (unsigned __int8)sub_BAA6E0() )
    return ((*(_BYTE *)(a1 + 33) >> 6) ^ 1) & 1;
  else
    return 0;
}
