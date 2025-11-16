// Function: sub_31DB810
// Address: 0x31db810
//
char __fastcall sub_31DB810(__int64 a1)
{
  __int64 v1; // rdx
  char result; // al

  v1 = *(_QWORD *)(a1 + 208);
  result = 0;
  if ( !*(_DWORD *)(v1 + 336) )
  {
    result = *(_BYTE *)(v1 + 340);
    if ( result )
      return *(_DWORD *)(a1 + 776) != 0;
  }
  return result;
}
