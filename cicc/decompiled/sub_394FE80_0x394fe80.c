// Function: sub_394FE80
// Address: 0x394fe80
//
bool __fastcall sub_394FE80(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, char a5)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  unsigned int v8; // r15d
  int v10; // r8d
  bool result; // al
  _QWORD *v12; // rdx
  _QWORD *v13; // rax

  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v6 = *(_QWORD *)(a2 + 24 * (a3 - v5));
  v7 = *(_QWORD *)(a2 + 24 * (a4 - v5));
  if ( v7 == v6 )
    return 1;
  if ( *(_BYTE *)(v6 + 16) != 13 )
    return 0;
  v8 = *(_DWORD *)(v6 + 32);
  if ( v8 > 0x40 )
  {
    v10 = sub_16A58F0(v6 + 24);
    result = 1;
    if ( v8 == v10 )
      return result;
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    goto LABEL_6;
  }
  result = 1;
  if ( *(_QWORD *)(v6 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) )
  {
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
LABEL_6:
    if ( a5 )
    {
      v13 = (_QWORD *)sub_14AD030(v7, 8u);
      if ( v13 )
      {
        v12 = *(_QWORD **)(v6 + 24);
        if ( *(_DWORD *)(v6 + 32) > 0x40u )
          v12 = (_QWORD *)*v12;
        return v13 <= v12;
      }
    }
    else if ( *(_BYTE *)(v7 + 16) == 13 )
    {
      v12 = *(_QWORD **)(v6 + 24);
      if ( v8 > 0x40 )
        v12 = (_QWORD *)*v12;
      v13 = *(_QWORD **)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v13 = (_QWORD *)*v13;
      return v13 <= v12;
    }
    return 0;
  }
  return result;
}
