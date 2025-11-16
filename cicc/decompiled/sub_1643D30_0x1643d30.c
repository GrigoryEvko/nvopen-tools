// Function: sub_1643D30
// Address: 0x1643d30
//
__int64 __fastcall sub_1643D30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // cc
  _QWORD *v4; // rax

  if ( *(_BYTE *)(a1 + 8) != 13 )
    return *(_QWORD *)(a1 + 24);
  v2 = sub_15A0FC0(a2);
  v3 = *(_DWORD *)(v2 + 8) <= 0x40u;
  v4 = *(_QWORD **)v2;
  if ( !v3 )
    v4 = (_QWORD *)*v4;
  return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * (unsigned int)v4);
}
