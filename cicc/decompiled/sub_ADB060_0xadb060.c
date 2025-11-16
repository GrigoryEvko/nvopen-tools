// Function: sub_ADB060
// Address: 0xadb060
//
__int64 __fastcall sub_ADB060(unsigned __int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // eax

  v2 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  v3 = a2;
  v4 = *(_DWORD *)(v2 + 8) >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(a2 + 16);
  if ( *(_DWORD *)(v3 + 8) >> 8 == v4 )
    return sub_AD4C90(a1, (__int64 **)a2, 0);
  else
    return sub_ADA8A0(a1, a2, 0);
}
