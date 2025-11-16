// Function: sub_C20C60
// Address: 0xc20c60
//
char __fastcall sub_C20C60(__int64 a1)
{
  char result; // al

  result = *(_BYTE *)(a1 + 178);
  if ( !result && !*(_BYTE *)(a1 + 204) )
    return *(_QWORD *)(a1 + 88) != 0;
  return result;
}
