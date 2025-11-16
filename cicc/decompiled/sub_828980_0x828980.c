// Function: sub_828980
// Address: 0x828980
//
__int64 __fastcall sub_828980(__int64 a1)
{
  char v1; // al
  __int64 result; // rax
  _BOOL4 v3; // r8d

  v1 = *(_BYTE *)(a1 + 176);
  if ( v1 != 1 )
    return v1 == 0;
  if ( (unsigned int)sub_8D32E0(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 120LL)) )
    return *(_BYTE *)(a1 + 176) == 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 184) + 170LL) & 1) != 0 )
    return *(_BYTE *)(a1 + 176) == 0;
  v3 = sub_730840(a1);
  result = 1;
  if ( !v3 )
    return *(_BYTE *)(a1 + 176) == 0;
  return result;
}
