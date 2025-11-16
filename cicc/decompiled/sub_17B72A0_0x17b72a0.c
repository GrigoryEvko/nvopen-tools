// Function: sub_17B72A0
// Address: 0x17b72a0
//
__int64 __fastcall sub_17B72A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 80);
  if ( v1 == a1 + 72 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    v2 = *(_QWORD *)(v1 + 24);
    if ( v2 != v1 + 16 )
      break;
LABEL_13:
    v1 = *(_QWORD *)(v1 + 8);
    if ( a1 + 72 == v1 )
      return 0;
  }
  while ( 1 )
  {
    if ( !v2 )
      BUG();
    if ( *(_BYTE *)(v2 - 8) != 78
      || (v4 = *(_QWORD *)(v2 - 48), *(_BYTE *)(v4 + 16))
      || (*(_BYTE *)(v4 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v4 + 36) - 35) > 3 )
    {
      if ( *(_QWORD *)(v2 + 24) && (unsigned int)sub_15C70B0(v2 + 24) )
        return 1;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( v1 + 16 == v2 )
      goto LABEL_13;
  }
}
