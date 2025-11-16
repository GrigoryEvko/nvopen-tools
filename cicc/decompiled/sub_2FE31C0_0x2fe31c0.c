// Function: sub_2FE31C0
// Address: 0x2fe31c0
//
bool __fastcall sub_2FE31C0(__int64 a1, unsigned __int16 a2, __int64 a3, unsigned __int16 a4, __int64 a5, bool a6)
{
  bool result; // al

  result = a6;
  if ( a6 )
  {
    result = 0;
    if ( a2 && *(_QWORD *)(a1 + 8LL * a2 + 112) )
    {
      if ( a4 )
        return *(_BYTE *)(a4 + 274LL * a2 + a1 + 443718) == 0;
    }
  }
  else if ( a2 && *(_QWORD *)(a1 + 8LL * a2 + 112) && a4 )
  {
    return (*(_BYTE *)(a4 + 274LL * a2 + a1 + 443718) & 0xFB) == 0;
  }
  return result;
}
