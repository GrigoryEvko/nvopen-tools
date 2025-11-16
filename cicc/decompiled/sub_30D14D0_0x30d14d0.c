// Function: sub_30D14D0
// Address: 0x30d14d0
//
__int64 __fastcall sub_30D14D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax

  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 )
    return 0;
  if ( !sub_AD0010(a2) )
    return 0;
  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 80);
  LOBYTE(v3) = *(_QWORD *)(v2 + 24) == v3;
  LOBYTE(v2) = a2 == v2;
  return (unsigned int)v2 & (unsigned int)v3;
}
