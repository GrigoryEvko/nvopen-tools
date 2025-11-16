// Function: sub_72FD90
// Address: 0x72fd90
//
__int64 __fastcall sub_72FD90(__int64 a1, char a2)
{
  __int64 result; // rax
  int v5; // esi
  int v6; // ecx
  int v7; // edi

  result = a1;
  if ( !a1 )
    return 0;
  v5 = a2 & 1;
  v6 = a2 & 4;
  v7 = a2 & 8;
  do
  {
    if ( (!v6 || *(_QWORD *)result)
      && (!v7 || (*(_BYTE *)(result + 146) & 8) == 0)
      && (!v5 || *(_QWORD *)(result + 8)
              || (*(_BYTE *)(result + 144) & 4) == 0
              || (*(_BYTE *)(result + 144) & 0x10) != 0) )
    {
      break;
    }
    result = *(_QWORD *)(result + 112);
  }
  while ( result );
  return result;
}
