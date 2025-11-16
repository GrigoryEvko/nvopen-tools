// Function: sub_1E9C0A0
// Address: 0x1e9c0a0
//
__int64 __fastcall sub_1E9C0A0(__int64 a1, int a2, __int16 a3)
{
  int v4; // ebx
  __int64 v5; // r12

  if ( *(_DWORD *)(a1 + 16) != 2 )
    return 0;
  v4 = a3 & 0xFFF;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  sub_1E310D0(v5 + 80, a2);
  *(_DWORD *)(v5 + 80) = *(_DWORD *)(v5 + 80) & 0xFFF000FF | (v4 << 8);
  return 1;
}
