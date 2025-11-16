// Function: sub_278BC90
// Address: 0x278bc90
//
__int64 __fastcall sub_278BC90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // eax
  __int64 v6; // r8

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)(a1 + 24);
  if ( v2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(v2 + 44) + 1);
    v5 = *(_DWORD *)(v2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = 0;
  if ( v5 < *(_DWORD *)(v3 + 32) )
    return *(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v4);
  return v6;
}
