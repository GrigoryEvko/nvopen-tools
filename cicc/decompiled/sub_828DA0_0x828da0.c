// Function: sub_828DA0
// Address: 0x828da0
//
__int64 __fastcall sub_828DA0(__int64 a1, __int64 a2)
{
  int v2; // edx
  int v3; // ecx
  char v4; // al
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  if ( v2 == 7 )
    return v3 == 4;
  if ( v3 == 7 )
  {
    if ( v2 == 4 )
      return 0xFFFFFFFFLL;
    return v3 == 4;
  }
  v4 = *(_BYTE *)(a1 + 12);
  if ( v4 != *(_BYTE *)(a2 + 12) )
    return v4 == 0 ? 1 : -1;
  if ( ((*(_BYTE *)(a2 + 85) ^ *(_BYTE *)(a1 + 85)) & 0x80u) != 0 && (!(_DWORD)qword_4F077B4 || qword_4F077A0 > 0x77EBu) )
    return ((*(unsigned __int8 *)(a1 + 85) >> 6) & 2u) - 1;
  if ( v2 < v3 )
    return 1;
  if ( v2 > v3 )
    return 0xFFFFFFFFLL;
  if ( !unk_4D0430C )
  {
    result = sub_827710(a1, a2);
    if ( (_DWORD)result )
      return result;
    v2 = *(_DWORD *)(a1 + 8);
  }
  if ( *(_QWORD *)(a1 + 48) == *(_QWORD *)(a2 + 48) )
    return sub_826E60((__int64 *)(a1 + 72), (__int64 *)(a2 + 72), 0, 0, 0);
  result = 0;
  if ( v2 != 4 )
    return sub_826E60((__int64 *)(a1 + 72), (__int64 *)(a2 + 72), 0, 0, 0);
  return result;
}
