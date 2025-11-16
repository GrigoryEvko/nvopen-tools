// Function: sub_2450400
// Address: 0x2450400
//
__int64 __fastcall sub_2450400(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax

  LOBYTE(v1) = sub_ED2B90(a1);
  v2 = v1;
  if ( (_BYTE)v1 )
    return v2;
  v4 = sub_BA91D0(a1, "EnableValueProfiling", 0x14u);
  if ( !v4 || *(_BYTE *)v4 != 1 )
    return v2;
  v5 = *(_QWORD *)(v4 + 136);
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  LOBYTE(v2) = v6 != 0;
  return v2;
}
