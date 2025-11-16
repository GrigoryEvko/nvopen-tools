// Function: sub_20C8260
// Address: 0x20c8260
//
__int64 __fastcall sub_20C8260(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v2; // al
  const char *v4; // rax
  __int64 v5; // rdx

  v1 = sub_1649C60(a1);
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 == 3 )
  {
    v4 = sub_1649960(v1);
    if ( v5 != 23 )
      return v1;
    if ( *(_QWORD *)v4 ^ 0x2E68652E6D766C6CLL | *((_QWORD *)v4 + 1) ^ 0x6C612E6863746163LL )
      return v1;
    if ( *((_DWORD *)v4 + 4) != 1635135084 )
      return v1;
    if ( *((_WORD *)v4 + 10) != 30060 )
      return v1;
    if ( v4[22] != 101 )
      return v1;
    v1 = *(_QWORD *)(v1 - 24);
    if ( *(_BYTE *)(v1 + 16) <= 3u )
      return v1;
  }
  else if ( v2 <= 2u )
  {
    return v1;
  }
  return 0;
}
