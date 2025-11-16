// Function: sub_14A8F90
// Address: 0x14a8f90
//
__int64 __fastcall sub_14A8F90(__int64 a1, unsigned int a2, unsigned int a3, char a4)
{
  __int64 v4; // rbx

  LODWORD(v4) = a3;
  if ( a4 )
  {
    v4 = (unsigned int)sub_15FF5D0(a3);
    if ( (unsigned __int8)sub_15FF880(a2, v4) )
      goto LABEL_3;
  }
  else if ( (unsigned __int8)sub_15FF880(a2, a3) )
  {
LABEL_3:
    *(_WORD *)a1 = 257;
    return a1;
  }
  if ( (unsigned __int8)sub_15FF910(a2, (unsigned int)v4) )
    *(_WORD *)a1 = 256;
  else
    *(_BYTE *)(a1 + 1) = 0;
  return a1;
}
