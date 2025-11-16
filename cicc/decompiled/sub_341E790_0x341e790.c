// Function: sub_341E790
// Address: 0x341e790
//
__int64 __fastcall sub_341E790(__int64 a1, __int64 a2)
{
  signed __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12

  v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3);
  v3 = a1;
  if ( v2 >> 2 > 0 )
  {
    v4 = a1 + 160 * (v2 >> 2);
    do
    {
      if ( (unsigned __int8)sub_B2D670(v3, 73) )
        return v3;
      v5 = v3 + 40;
      if ( (unsigned __int8)sub_B2D670(v3 + 40, 73) )
        return v5;
      v5 = v3 + 80;
      if ( (unsigned __int8)sub_B2D670(v3 + 80, 73) )
        return v5;
      v5 = v3 + 120;
      if ( (unsigned __int8)sub_B2D670(v3 + 120, 73) )
        return v5;
      v3 += 160;
    }
    while ( v3 != v4 );
    v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v3) >> 3);
  }
  if ( v2 == 2 )
  {
LABEL_19:
    v5 = v3;
    if ( !(unsigned __int8)sub_B2D670(v3, 73) )
    {
      v3 += 40;
      goto LABEL_14;
    }
    return v5;
  }
  if ( v2 == 3 )
  {
    v5 = v3;
    if ( (unsigned __int8)sub_B2D670(v3, 73) )
      return v5;
    v3 += 40;
    goto LABEL_19;
  }
  v5 = a2;
  if ( v2 != 1 )
    return v5;
LABEL_14:
  if ( !(unsigned __int8)sub_B2D670(v3, 73) )
    return a2;
  return v3;
}
