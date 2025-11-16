// Function: sub_5E69C0
// Address: 0x5e69c0
//
__int64 __fastcall sub_5E69C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rdx

  if ( (*(_BYTE *)(a1 + 17) & 0x40) == 0 )
  {
    *(_BYTE *)(a1 + 16) &= ~0x80u;
    *(_QWORD *)(a1 + 24) = 0;
  }
  sub_7D2AC0(a1, a2, 4096);
  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    v3 = *(unsigned __int8 *)(result + 80);
    if ( (unsigned __int8)v3 > 0x14u )
      return 0;
    v4 = 1182720;
    if ( _bittest64(&v4, v3) )
      return result;
    if ( (_BYTE)v3 != 16 || (*(_BYTE *)(result + 96) & 4) == 0 )
      return 0;
    v5 = **(_QWORD **)(result + 88);
    v6 = *(unsigned __int8 *)(v5 + 80);
    if ( (_BYTE)v6 == 24 )
    {
      v5 = *(_QWORD *)(v5 + 88);
      v6 = *(unsigned __int8 *)(v5 + 80);
    }
    if ( (unsigned __int8)v6 > 0x14u )
      return 0;
    v7 = 1182720;
    if ( !_bittest64(&v7, v6) )
    {
      if ( (_BYTE)v6 != 2 )
        return 0;
      v8 = *(_QWORD *)(v5 + 88);
      if ( !v8 )
        return 0;
      if ( *(_BYTE *)(v8 + 173) != 12 )
        return 0;
    }
  }
  return result;
}
