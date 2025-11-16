// Function: sub_23FAE50
// Address: 0x23fae50
//
__int64 __fastcall sub_23FAE50(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned int v5; // eax
  unsigned int v6; // r8d

  v2 = *(_QWORD *)(*a1 + 16LL);
  v3 = *(_QWORD *)(*(_QWORD *)(a1[1] - 8LL) + 32LL * *(unsigned int *)(a1[1] + 72LL) + 8LL * a2);
  if ( v3 )
  {
    v4 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
    v5 = *(_DWORD *)(v3 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = 1;
  if ( v5 < *(_DWORD *)(v2 + 32) )
    LOBYTE(v6) = *(_QWORD *)(*(_QWORD *)(v2 + 24) + 8 * v4) == 0;
  return v6;
}
