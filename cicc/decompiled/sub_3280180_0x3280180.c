// Function: sub_3280180
// Address: 0x3280180
//
bool __fastcall sub_3280180(__int64 a1)
{
  __int16 v1; // ax
  bool v2; // dl

  v1 = *(_WORD *)a1;
  if ( !*(_WORD *)a1 )
    return sub_3007070(a1);
  v2 = (unsigned __int16)(v1 - 17) <= 0x6Cu || (unsigned __int16)(v1 - 2) <= 7u;
  if ( !v2 )
    return (unsigned __int16)(v1 - 176) <= 0x1Fu;
  return v2;
}
