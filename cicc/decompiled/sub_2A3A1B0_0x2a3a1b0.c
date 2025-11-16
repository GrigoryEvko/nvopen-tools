// Function: sub_2A3A1B0
// Address: 0x2a3a1b0
//
__int64 __fastcall sub_2A3A1B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int16 v7; // ax
  __int64 *v8; // rdi

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(unsigned __int8 *)(v3 + 8);
  if ( (unsigned __int8)v4 > 0xCu || (v5 = 4143, !_bittest64(&v5, v4)) )
  {
    if ( (v4 & 0xFB) != 0xA && (v4 & 0xFD) != 4 )
    {
      if ( (unsigned __int8)(v4 - 15) > 3u && (_BYTE)v4 != 20 || !(unsigned __int8)sub_BCEBA0(v3, 0) )
        return 0;
      v3 = *(_QWORD *)(a2 + 72);
    }
  }
  if ( sub_BCEA30(v3) )
    return 0;
  if ( !sub_B4D040(a2) )
    return 0;
  if ( !sub_2A3A050(a2) )
    return 0;
  if ( (unsigned __int8)sub_2A4D8A0(a2, 0) )
    return 0;
  v7 = *(_WORD *)(a2 + 2);
  if ( (v7 & 0x40) != 0 || (v7 & 0x80u) != 0 )
    return 0;
  v8 = *(__int64 **)(a1 + 184);
  if ( v8 && (unsigned __int8)sub_D90430(v8, a2) )
    return 1;
  else
    return 2;
}
