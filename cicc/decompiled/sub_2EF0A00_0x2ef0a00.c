// Function: sub_2EF0A00
// Address: 0x2ef0a00
//
__int64 __fastcall sub_2EF0A00(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 result; // rax

  v2 = **(_QWORD **)a1;
  if ( a2 >= (*(_DWORD *)(v2 + 40) & 0xFFFFFFu) )
    return (__int64)sub_2EF06E0(*(_QWORD *)(a1 + 8), "stack map constant to STATEPOINT is out of range!", v2);
  v3 = *(_QWORD *)(v2 + 32);
  v4 = v3 + 40LL * (a2 - 1);
  if ( *(_BYTE *)v4 != 1 )
    return (__int64)sub_2EF06E0(*(_QWORD *)(a1 + 8), "stack map constant to STATEPOINT not well formed!", v2);
  if ( *(_QWORD *)(v4 + 24) != 2 )
    return (__int64)sub_2EF06E0(*(_QWORD *)(a1 + 8), "stack map constant to STATEPOINT not well formed!", v2);
  result = 5LL * a2;
  if ( *(_BYTE *)(v3 + 40LL * a2) != 1 )
    return (__int64)sub_2EF06E0(*(_QWORD *)(a1 + 8), "stack map constant to STATEPOINT not well formed!", v2);
  return result;
}
