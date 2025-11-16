// Function: sub_250ED40
// Address: 0x250ed40
//
__int64 __fastcall sub_250ED40(__int64 a1)
{
  __int64 v2; // [rsp+0h] [rbp-8h]

  if ( ((*(_DWORD *)(a1 + 376) - 26) & 0xFFFFFFEE) != 0 )
  {
    BYTE4(v2) = 0;
  }
  else
  {
    LODWORD(v2) = 0;
    BYTE4(v2) = 1;
  }
  return v2;
}
