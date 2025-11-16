// Function: sub_2DDC600
// Address: 0x2ddc600
//
__int64 __fastcall sub_2DDC600(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rbx

  if ( sub_B2FC80(a1)
    || (unsigned __int8)sub_B2D610(a1, 32)
    || (unsigned __int8)sub_B2D610(a1, 3)
    || (*(_BYTE *)(a1 + 32) & 0xF) == 1
    || *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8
    || ((*(_WORD *)(a1 + 2) >> 4) & 0x3FF) == 0x14 )
  {
    return 0;
  }
  v2 = *(_QWORD *)(a1 + 80);
  v3 = 0x8000000000041LL;
  if ( v2 != a1 + 72 )
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      v4 = *(_QWORD *)(v2 + 32);
      if ( v4 != v2 + 24 )
        break;
LABEL_16:
      v2 = *(_QWORD *)(v2 + 8);
      if ( a1 + 72 == v2 )
        return 1;
    }
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v4 - 24) - 34) <= 0x33u
        && _bittest64(&v3, (unsigned int)*(unsigned __int8 *)(v4 - 24) - 34)
        && sub_B49200(v4 - 24) )
      {
        return 0;
      }
      v4 = *(_QWORD *)(v4 + 8);
      if ( v2 + 24 == v4 )
        goto LABEL_16;
    }
  }
  return 1;
}
