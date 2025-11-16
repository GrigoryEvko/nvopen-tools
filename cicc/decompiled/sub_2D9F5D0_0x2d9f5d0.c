// Function: sub_2D9F5D0
// Address: 0x2d9f5d0
//
bool __fastcall sub_2D9F5D0(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // r8
  bool result; // al
  int v6; // ecx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL);
  v3 = v2;
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v2 + 16);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v1 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v1 + 16);
  result = 0;
  if ( v4 == v3 )
  {
    v6 = 2 * *(_DWORD *)(v1 + 32);
    if ( v6 == *(_DWORD *)(v2 + 32) )
      return *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned int *)(a1 + 80) - 4) < v6;
  }
  return result;
}
