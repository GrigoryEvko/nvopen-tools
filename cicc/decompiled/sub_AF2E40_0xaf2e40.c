// Function: sub_AF2E40
// Address: 0xaf2e40
//
__int64 __fastcall sub_AF2E40(__int64 a1)
{
  __int64 v2; // [rsp+8h] [rbp-18h]

  if ( (unsigned __int16)sub_AF18C0(a1) == 17152 )
  {
    BYTE4(v2) = 1;
    LODWORD(v2) = *(_DWORD *)(a1 + 4);
  }
  else
  {
    BYTE4(v2) = 0;
  }
  return v2;
}
