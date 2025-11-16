// Function: sub_31DA250
// Address: 0x31da250
//
__int64 __fastcall sub_31DA250(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // eax
  unsigned __int16 v5; // ax
  unsigned int v6; // r8d
  int v7; // edx

  v3 = 0;
  if ( *(_BYTE *)a1 == 3 )
    LOBYTE(v3) = sub_AE5270(a2, a1);
  if ( (unsigned __int8)a3 < (unsigned __int8)v3 )
    a3 = v3;
  v5 = *(_WORD *)(a1 + 34) >> 1;
  v6 = a3;
  v7 = v5;
  LOWORD(v7) = v5 & 0x3F;
  if ( (v5 & 0x3F) == 0 )
    return v6;
  v6 = v7 - 1;
  if ( (unsigned __int8)a3 < (unsigned __int8)(v7 - 1) )
    return v6;
  if ( (v5 & 0x200) == 0 )
    return a3;
  return v6;
}
