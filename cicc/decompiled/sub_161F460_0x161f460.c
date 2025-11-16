// Function: sub_161F460
// Address: 0x161f460
//
__int64 __fastcall sub_161F460(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi
  _QWORD *v5; // rdx
  _QWORD *v6; // rcx

  result = a2;
  if ( !a1 || !a2 )
    return 0;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8)) + 136LL);
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)) + 136LL);
  v5 = *(_QWORD **)(v3 + 24);
  if ( *(_DWORD *)(v3 + 32) > 0x40u )
    v5 = (_QWORD *)*v5;
  v6 = *(_QWORD **)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  if ( v6 > v5 )
    return a1;
  return result;
}
