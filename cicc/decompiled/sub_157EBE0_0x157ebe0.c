// Function: sub_157EBE0
// Address: 0x157ebe0
//
unsigned __int64 __fastcall sub_157EBE0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rcx
  unsigned __int64 v4; // r8
  __int64 v5; // rax

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v1 == (_QWORD *)(a1 + 40) )
    return 0;
  if ( !v1 )
    BUG();
  if ( *((_BYTE *)v1 - 8) != 25 )
    return 0;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 )
  {
    if ( v1 - 3 == (_QWORD *)(v2 - 24) )
      return 0;
  }
  if ( *(_QWORD **)(v1[2] + 48LL) == v1 )
    return 0;
  v3 = (_QWORD *)(*v1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v3 )
    return 0;
  v4 = (unsigned __int64)(v3 - 3);
  if ( (*((_DWORD *)v1 - 1) & 0xFFFFFFF) != 0 )
  {
    v5 = v1[-3 * (*((_DWORD *)v1 - 1) & 0xFFFFFFF) - 3];
    if ( v5 )
    {
      if ( v5 != v4 )
        return 0;
      if ( *((_BYTE *)v3 - 8) == 71 )
      {
        if ( *(_QWORD **)(v3[2] + 48LL) == v3 )
          return 0;
        if ( (*v3 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          return 0;
        v4 = (*v3 & 0xFFFFFFFFFFFFFFF8LL) - 24;
        if ( *(v3 - 6) != v4 )
          return 0;
      }
    }
  }
  if ( *(_BYTE *)(v4 + 16) != 78 || (*(_WORD *)(v4 + 18) & 3) != 2 )
    return 0;
  return v4;
}
