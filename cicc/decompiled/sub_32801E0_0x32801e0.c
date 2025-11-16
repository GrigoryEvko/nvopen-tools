// Function: sub_32801E0
// Address: 0x32801e0
//
bool __fastcall sub_32801E0(__int64 a1)
{
  if ( *(_WORD *)a1 )
    return (unsigned __int16)(*(_WORD *)a1 - 17) <= 0xD3u;
  else
    return sub_30070B0(a1);
}
