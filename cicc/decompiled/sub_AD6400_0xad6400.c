// Function: sub_AD6400
// Address: 0xad6400
//
__int64 __fastcall sub_AD6400(__int64 a1)
{
  unsigned __int8 *v1; // rsi
  int v2; // edx
  __int64 v4; // [rsp+8h] [rbp-18h]

  v1 = (unsigned __int8 *)sub_ACD6D0(*(__int64 **)a1);
  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v2 - 17) > 1 )
    return (__int64)v1;
  BYTE4(v4) = (_BYTE)v2 == 18;
  LODWORD(v4) = *(_DWORD *)(a1 + 32);
  return sub_AD5E10(v4, v1);
}
