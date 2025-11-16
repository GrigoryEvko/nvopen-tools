// Function: sub_32801C0
// Address: 0x32801c0
//
bool __fastcall sub_32801C0(__int64 a1)
{
  if ( *(_WORD *)a1 )
    return (unsigned __int16)(*(_WORD *)a1 - 2) <= 7u;
  else
    return sub_30070A0(a1);
}
