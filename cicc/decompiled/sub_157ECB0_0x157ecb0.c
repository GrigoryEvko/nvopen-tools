// Function: sub_157ECB0
// Address: 0x157ecb0
//
unsigned __int64 __fastcall sub_157ECB0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rdx

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v1 == (_QWORD *)(a1 + 40) )
    return 0;
  if ( !v1 )
    BUG();
  if ( *((_BYTE *)v1 - 8) != 25 )
    return 0;
  v2 = *(_QWORD **)(a1 + 48);
  if ( v2 )
  {
    if ( v1 == v2 )
      return 0;
  }
  if ( *(_QWORD **)(v1[2] + 48LL) == v1 )
    return 0;
  v3 = *v1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v3 )
    return 0;
  if ( *(_BYTE *)(v3 - 8) != 78 )
    return 0;
  v4 = *(_QWORD *)(v3 - 48);
  if ( *(_BYTE *)(v4 + 16) || *(_DWORD *)(v4 + 36) != 75 )
    return 0;
  else
    return v3 - 24;
}
