// Function: sub_39CB4A0
// Address: 0x39cb4a0
//
__int64 __fastcall sub_39CB4A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // r8d
  __int64 v6; // r9

  v2 = 0;
  if ( sub_39C7D40((__int64)a1, a2) )
    return v2;
  v3 = sub_145CDC0(0x30u, a1 + 11);
  v2 = v3;
  if ( v3 )
  {
    *(_BYTE *)(v3 + 30) = 0;
    *(_QWORD *)v3 = v3 | 4;
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = 0;
    *(_DWORD *)(v3 + 24) = -1;
    *(_WORD *)(v3 + 28) = 11;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
  }
  if ( *(_BYTE *)(a2 + 24) )
    return v2;
  sub_39CB2D0(a1, v3, a2 + 80, v4, v5, v6);
  return v2;
}
