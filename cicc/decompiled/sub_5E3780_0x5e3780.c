// Function: sub_5E3780
// Address: 0x5e3780
//
__int64 __fastcall sub_5E3780(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13

  if ( dword_4F04C58 == -1 )
  {
    result = unk_4F04C50;
    if ( !unk_4F04C50 )
      return result;
    v3 = *(_QWORD *)(unk_4F04C50 + 32LL);
  }
  else
  {
    result = unk_4F04C68 + 776LL * dword_4F04C58;
    v3 = *(_QWORD *)(result + 216);
  }
  if ( v3 && a1 )
  {
    if ( (*(_BYTE *)(v3 + 193) & 0x10) == 0
      && ((result = *(_QWORD *)(v3 + 200) & 0x8000001000000LL, result != 0x8000000000000LL)
       || (*(_BYTE *)(v3 + 192) & 2) != 0)
      || (*(_BYTE *)(v3 + 197) & 0x60) == 0 && (*(_BYTE *)(v3 + 198) & 0x18) != 0 )
    {
      if ( *(_BYTE *)(a1 + 174) || !*(_WORD *)(a1 + 176) )
      {
        if ( (*(_BYTE *)(a1 + 193) & 0x10) == 0 )
        {
          result = *(_QWORD *)(a1 + 200) & 0x8000001000000LL;
          if ( result != 0x8000000000000LL || (*(_BYTE *)(a1 + 192) & 2) != 0 )
            goto LABEL_24;
        }
        result = *(_BYTE *)(a1 + 205) & 0x1C;
        if ( (_BYTE)result == 8 )
          goto LABEL_24;
      }
      else
      {
        result = sub_825D70();
        if ( !(_DWORD)result )
          goto LABEL_24;
      }
      result = *(unsigned int *)(a1 + 160);
      if ( !(_DWORD)result || (*(_BYTE *)(a1 + 197) & 0x60) != 0 || (*(_BYTE *)(a1 + 198) & 0x18) == 0 )
        return result;
LABEL_24:
      if ( (*(_BYTE *)(v3 + 198) & 0x10) != 0 )
        return sub_825950(v3, a1, 1, a2);
    }
  }
  return result;
}
