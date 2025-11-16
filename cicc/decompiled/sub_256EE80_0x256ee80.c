// Function: sub_256EE80
// Address: 0x256ee80
//
__int64 __fastcall sub_256EE80(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rbx
  __int64 v5; // r12
  unsigned __int8 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  v2 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v2 = *(_QWORD *)(v2 + 24);
  v3 = (unsigned int)sub_250CB50((__int64 *)(a1 + 72), 0);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(v2 - 8);
  else
    v4 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v5 = v4 + 32 * v3;
  v6 = (unsigned __int8 *)sub_ACA8A0(*(__int64 ***)(*(_QWORD *)v5 + 8LL));
  return (unsigned __int8)sub_256E5A0(a2, v5, v6, v7, v8, v9) ^ 1u;
}
