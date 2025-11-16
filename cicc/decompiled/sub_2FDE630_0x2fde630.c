// Function: sub_2FDE630
// Address: 0x2fde630
//
__int64 __fastcall sub_2FDE630(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // r8

  v2 = a2;
  v3 = *(_DWORD *)(a2 + 44);
  if ( (v3 & 4) == 0 && (v3 & 8) != 0 )
  {
    a2 = 128;
    if ( sub_2E88A90(v2, 128, 1) )
      return 1;
  }
  else if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x80u) != 0LL )
  {
    return 1;
  }
  if ( sub_2E8B090(v2) )
    return 1;
  result = sub_2E8B100(v2, a2, v5, v6, v7);
  if ( (_BYTE)result )
    return (unsigned int)sub_2E8AED0(v2) ^ 1;
  return result;
}
