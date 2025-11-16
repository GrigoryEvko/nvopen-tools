// Function: sub_BAA6E0
// Address: 0xbaa6e0
//
bool __fastcall sub_BAA6E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rax

  v1 = sub_BA91D0(a1, "SemanticInterposition", 0x15u);
  if ( !v1 )
    return 0;
  v2 = *(_QWORD *)(v1 + 136);
  v3 = *(_QWORD **)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    v3 = (_QWORD *)*v3;
  return v3 != 0;
}
