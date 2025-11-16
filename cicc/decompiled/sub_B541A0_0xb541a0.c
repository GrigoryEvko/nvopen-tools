// Function: sub_B541A0
// Address: 0xb541a0
//
__int64 __fastcall sub_B541A0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax

  if ( *(_BYTE *)(a1 + 56) )
  {
    v3 = *(unsigned int *)(a1 + 16);
    v4 = *(_QWORD *)(a1 + 8);
    *(_BYTE *)(a1 + 64) = 1;
    *(_DWORD *)(v4 + 4LL * (unsigned int)(a3 + 1)) = *(_DWORD *)(v4 + 4 * v3 - 4);
    --*(_DWORD *)(a1 + 16);
  }
  return sub_B53C80(*(_QWORD *)a1, a2, a3);
}
