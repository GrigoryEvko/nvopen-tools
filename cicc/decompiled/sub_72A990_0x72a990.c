// Function: sub_72A990
// Address: 0x72a990
//
_BOOL8 __fastcall sub_72A990(__int64 a1)
{
  _BOOL8 result; // rax
  unsigned __int8 v2; // r12
  __int64 v3; // rax
  int v4; // eax
  __int64 v5; // rdi

  if ( *(_BYTE *)(a1 + 173) != 6 )
    return 0;
  v2 = *(_BYTE *)(a1 + 176);
  v3 = sub_72A940(a1);
  if ( v2 == 5 )
    return 0;
  if ( v2 > 5u )
  {
    if ( v2 != 6 )
      sub_721090();
  }
  else if ( !v2 && unk_4D0480C )
  {
    return 0;
  }
  if ( !v3 )
    return 0;
  if ( (*(_BYTE *)(v3 + 89) & 4) != 0 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + 32LL);
    result = 1;
    if ( (*(_BYTE *)(v5 + 89) & 1) == 0 )
    {
      sub_8DCD50();
      return 0;
    }
  }
  else
  {
    v4 = *(_BYTE *)(v3 + 88) & 0x70;
    if ( v4 )
    {
      if ( (_BYTE)v4 == 16 )
        return unk_4D04440 == 0;
      return 0;
    }
    result = 1;
    if ( unk_4D0480C && (!v2 || v2 == 1 && *(_DWORD *)(*(_QWORD *)(a1 + 184) + 64LL)) )
      return 0;
  }
  return result;
}
