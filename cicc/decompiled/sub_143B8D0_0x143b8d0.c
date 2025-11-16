// Function: sub_143B8D0
// Address: 0x143b8d0
//
__int64 __fastcall sub_143B8D0(_QWORD *a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rbx
  unsigned __int8 v3; // al
  __int64 v4; // rbx

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 + 16LL);
  if ( v3 <= 0x17u )
    return 1;
  LOBYTE(v1) = v3 == 56 || v3 == 77;
  if ( (_BYTE)v1 )
    return 1;
  if ( (unsigned int)v3 - 60 > 0xC )
    goto LABEL_4;
  if ( (unsigned __int8)sub_14AF470(*a1, 0, 0, 0) )
    return 1;
  v3 = *(_BYTE *)(v2 + 16);
LABEL_4:
  if ( v3 != 35 )
    return v1;
  if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(v2 - 8);
  else
    v4 = v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF);
  LOBYTE(v1) = *(_BYTE *)(*(_QWORD *)(v4 + 24) + 16LL) == 13;
  return v1;
}
