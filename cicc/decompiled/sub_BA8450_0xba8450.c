// Function: sub_BA8450
// Address: 0xba8450
//
unsigned __int64 __fastcall sub_BA8450(__int64 a1)
{
  _BYTE *v1; // r12
  unsigned int v3; // ebx
  __int64 v4; // rsi
  unsigned __int8 v5; // cl
  __int64 v6; // r13

  v1 = *(_BYTE **)(a1 + 136);
  if ( !v1 || *v1 != 15 || !(unsigned int)sub_AC5290(*(_QWORD *)(a1 + 136)) )
    return 0;
  v3 = sub_AC5320((__int64)v1, 0);
  if ( (unsigned int)sub_AC5290((__int64)v1) > 1 )
  {
    v6 = sub_AC5320((__int64)v1, 1u) & 0x7FFFFFFF;
    v4 = 0;
    if ( (unsigned int)sub_AC5290((__int64)v1) > 2 )
      v4 = sub_AC5320((__int64)v1, 2u) & 0x7FFFFFFF;
    v5 = 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
    v6 = 0;
  }
  return (((v4 << 32) | v6 | ((unsigned __int64)v5 << 31)) << 32) | v3;
}
