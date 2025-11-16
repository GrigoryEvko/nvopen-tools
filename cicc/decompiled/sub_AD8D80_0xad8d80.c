// Function: sub_AD8D80
// Address: 0xad8d80
//
__int64 __fastcall sub_AD8D80(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rsi
  int v3; // edx
  __int64 v5; // [rsp+8h] [rbp-18h]

  v2 = (unsigned __int8 *)sub_ACCFD0(*(__int64 **)a1, a2);
  v3 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v3 - 17) > 1 )
    return (__int64)v2;
  BYTE4(v5) = (_BYTE)v3 == 18;
  LODWORD(v5) = *(_DWORD *)(a1 + 32);
  return sub_AD5E10(v5, v2);
}
