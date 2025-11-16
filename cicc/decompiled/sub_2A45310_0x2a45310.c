// Function: sub_2A45310
// Address: 0x2a45310
//
bool __fastcall sub_2A45310(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = *(_BYTE *)a1 > 0x1Cu || *(_BYTE *)a1 == 22;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 16);
    if ( v2 )
      return *(_QWORD *)(v2 + 8) != 0;
  }
  return result;
}
