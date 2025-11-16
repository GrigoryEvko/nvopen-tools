// Function: sub_667110
// Address: 0x667110
//
__int64 __fastcall sub_667110(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // rcx
  char v4; // dl

  result = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return sub_6851C0(771, a1 + 32);
  v2 = *(_BYTE *)(result + 80);
  v3 = *(_QWORD *)a1;
  if ( v2 == 20 )
  {
    v3 = **(_QWORD **)(*(_QWORD *)(result + 88) + 176LL);
    if ( !v3 )
      return sub_6851C0(771, a1 + 32);
    if ( (unsigned __int8)(*(_BYTE *)(v3 + 80) - 10) > 1u )
      goto LABEL_4;
LABEL_8:
    if ( (*(_BYTE *)(a1 + 8) & 8) == 0 )
    {
      v4 = *(_BYTE *)(*(_QWORD *)(v3 + 88) + 174LL);
      if ( v4 == 7 || (v4 == 1 || v4 == 3 && unk_4D044AC) && (*(_BYTE *)(a1 + 121) & 0x40) != 0 )
        return result;
    }
    goto LABEL_4;
  }
  if ( (unsigned __int8)(v2 - 10) <= 1u )
    goto LABEL_8;
LABEL_4:
  if ( (*(_BYTE *)(v3 + 81) & 0x20) == 0 && (*(_BYTE *)(result + 81) & 0x20) == 0 )
    return sub_6851C0(771, a1 + 32);
  return result;
}
