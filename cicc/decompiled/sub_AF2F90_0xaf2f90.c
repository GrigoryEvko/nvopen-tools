// Function: sub_AF2F90
// Address: 0xaf2f90
//
__int64 __fastcall sub_AF2F90(__int64 a1, __int64 a2, int a3)
{
  int v4; // edx
  __int64 v5; // [rsp+0h] [rbp-8h]

  if ( a2 == 7 )
  {
    if ( *(_DWORD *)a1 == 1598772035 && *(_WORD *)(a1 + 4) == 17485 && *(_BYTE *)(a1 + 6) == 53 )
    {
      v4 = 1;
      goto LABEL_7;
    }
LABEL_3:
    LODWORD(v5) = a3;
    BYTE4(v5) = 0;
    return v5;
  }
  if ( a2 == 8 )
  {
    if ( *(_QWORD *)a1 != 0x314148535F4B5343LL )
      goto LABEL_3;
    v4 = 2;
  }
  else
  {
    if ( a2 != 10 || *(_QWORD *)a1 != 0x324148535F4B5343LL || *(_WORD *)(a1 + 8) != 13877 )
      goto LABEL_3;
    v4 = 3;
  }
LABEL_7:
  LODWORD(v5) = v4;
  BYTE4(v5) = 1;
  return v5;
}
