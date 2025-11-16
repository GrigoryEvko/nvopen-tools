// Function: sub_1601EA0
// Address: 0x1601ea0
//
__int64 __fastcall sub_1601EA0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v3; // rax
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v1 + 16) )
    BUG();
  if ( *(_DWORD *)(v1 + 36) == 111 )
    return *(_QWORD *)(a1 + 24 * (4LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  v3 = (_QWORD *)sub_15F2050(a1);
  v4 = sub_1643360(*v3);
  return sub_159C470(v4, 1, 0);
}
