// Function: sub_10390E0
// Address: 0x10390e0
//
__int64 __fastcall sub_10390E0(__int64 a1)
{
  unsigned __int8 v1; // al

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    return **(_QWORD **)(a1 - 32);
  else
    return *(_QWORD *)(a1 - 8LL * ((v1 >> 2) & 0xF) - 16);
}
