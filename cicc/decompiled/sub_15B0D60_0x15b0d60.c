// Function: sub_15B0D60
// Address: 0x15b0d60
//
__int64 __fastcall sub_15B0D60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v4; // edx

  result = a1;
  if ( a3 == 7 )
  {
    if ( *(_DWORD *)a2 == 1598772035 && *(_WORD *)(a2 + 4) == 17485 )
    {
      v4 = 1;
      if ( *(_BYTE *)(a2 + 6) == 53 )
        goto LABEL_10;
    }
LABEL_3:
    *(_BYTE *)(a1 + 4) = 0;
    return result;
  }
  if ( a3 != 8 || *(_QWORD *)a2 != 0x314148535F4B5343LL )
    goto LABEL_3;
  v4 = 2;
LABEL_10:
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = v4;
  return result;
}
