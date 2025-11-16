// Function: sub_8C3930
// Address: 0x8c3930
//
_QWORD *__fastcall sub_8C3930(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 *v4; // r14

  if ( (*(_BYTE *)(a1 + 141) & 0x40) != 0 )
    *(_BYTE *)(a2 + 141) |= 0x40u;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    v2 = *(_QWORD *)(a2 + 168);
    if ( !*(_QWORD *)(v2 + 184) )
    {
      v4 = *(__int64 **)(*(_QWORD *)(a1 + 168) + 184LL);
      if ( v4 )
      {
        if ( (*(_BYTE *)(v4 - 1) & 3) == 3 )
        {
          sub_8C3650(v4, 0xBu, 0);
          v4 = (__int64 *)*(v4 - 3);
          if ( (*(_BYTE *)(v4 - 1) & 2) != 0 )
            v4 = (__int64 *)*(v4 - 3);
        }
        *(_QWORD *)(v2 + 184) = v4;
      }
    }
  }
  return sub_8C2B90(a1, a2);
}
