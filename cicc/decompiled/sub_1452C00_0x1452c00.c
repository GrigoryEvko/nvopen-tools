// Function: sub_1452C00
// Address: 0x1452c00
//
__int64 __fastcall sub_1452C00(__int64 a1)
{
  int v1; // edx
  __int64 v2; // rcx
  __int64 v4; // rsi

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)(v1 - 35) > 0x2Cu )
    return 0;
  v2 = 0x133FFE2BFFFFLL;
  if ( _bittest64(&v2, (unsigned int)(v1 - 35)) )
    return 1;
  if ( (_BYTE)v1 == 78 && (v4 = *(_QWORD *)(a1 - 24), !*(_BYTE *)(v4 + 16)) )
    return sub_14D90D0(a1 | 4, v4);
  else
    return 0;
}
