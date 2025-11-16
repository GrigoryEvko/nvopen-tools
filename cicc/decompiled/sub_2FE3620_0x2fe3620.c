// Function: sub_2FE3620
// Address: 0x2fe3620
//
bool __fastcall sub_2FE3620(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  bool result; // al

  if ( a5 == 1 || (result = 0, a5) && *(_QWORD *)(a1 + 8LL * a5 + 112) )
  {
    result = 1;
    if ( a2 <= 0x1F3 )
      return (*(_BYTE *)(a2 + 500LL * a5 + a1 + 6414) & 0xFB) == 0;
  }
  return result;
}
