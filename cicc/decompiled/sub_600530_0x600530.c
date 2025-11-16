// Function: sub_600530
// Address: 0x600530
//
__int64 __fastcall sub_600530(__int64 a1)
{
  __int64 i; // rax
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  char v6; // al
  __int64 result; // rax
  char v8; // dl

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  v3 = sub_5EB340(v2);
  v4 = v3;
  if ( !v3 )
    return 0;
  v5 = *(_QWORD *)(v3 + 88);
  v6 = *(_BYTE *)(v5 + 193);
  if ( (v6 & 0x10) == 0 && !(v6 & 2 | *(_BYTE *)(v5 + 206) & 8) )
    return 0;
  result = sub_6009B0(v5, a1, 1);
  if ( (_DWORD)result )
  {
    if ( (*(_BYTE *)(v5 + 193) & 0x10) != 0 || (*(_QWORD *)(v5 + 200) & 0x8000001000000LL) == 0x8000000000000LL )
    {
      *(_BYTE *)(v5 + 193) |= 2u;
      *(_BYTE *)(v2 + 183) |= 8u;
    }
  }
  else
  {
    if ( (*(_BYTE *)(v5 + 206) & 8) == 0 )
      return 0;
    v8 = *(_BYTE *)(v5 + 193);
    if ( (v8 & 2) == 0 )
      return 0;
    if ( (*(_BYTE *)(v5 + 195) & 3) != 1 && ((v8 & 4) != 0) | v8 & 1 )
    {
      sub_6851C0(2422, v4 + 48);
      result = 0;
    }
    *(_BYTE *)(v5 + 193) &= ~2u;
  }
  return result;
}
