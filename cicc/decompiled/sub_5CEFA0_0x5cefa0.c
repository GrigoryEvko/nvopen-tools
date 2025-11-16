// Function: sub_5CEFA0
// Address: 0x5cefa0
//
__int64 __fastcall sub_5CEFA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 10) - 2) <= 1u )
    goto LABEL_4;
  if ( unk_4F077C4 != 2 || !(unsigned int)sub_8D3A70(a2) && !(unsigned int)sub_8D2870(a2) )
  {
    if ( *(_BYTE *)(a2 + 140) != 7 )
    {
      result = sub_5CEF40(a2, 0);
      *(_BYTE *)(result + 143) |= 2u;
      return result;
    }
LABEL_4:
    *(_BYTE *)(a2 + 143) |= 2u;
    return a2;
  }
  sub_5CCAE0(8u, a1);
  return a2;
}
