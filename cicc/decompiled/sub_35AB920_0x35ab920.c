// Function: sub_35AB920
// Address: 0x35ab920
//
__int64 __fastcall sub_35AB920(__int64 a1)
{
  __int16 *v1; // rax
  __int16 *v2; // rdx
  int v3; // eax
  __int16 v4; // cx
  int v5; // eax

  v1 = *(__int16 **)(a1 + 8);
  if ( *(__int16 **)(a1 + 56) == v1 )
    return 0;
  v2 = v1 + 1;
  *(_QWORD *)(a1 + 8) = v1 + 1;
  v3 = *v1;
  v4 = v3;
  v5 = *(_DWORD *)a1 + v3;
  *(_DWORD *)a1 = v5;
  if ( !v4 )
    v2 = 0;
  *(_WORD *)(a1 + 16) = v5;
  *(_QWORD *)(a1 + 8) = v2;
  return 1;
}
