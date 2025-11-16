// Function: sub_34C78E0
// Address: 0x34c78e0
//
__int64 __fastcall sub_34C78E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  int v6; // eax
  unsigned int v7; // r12d

  v2 = **(_QWORD **)(a1 + 24);
  v3 = *(unsigned int *)(a2 + 112);
  if ( (int)v3 < 0 )
    v4 = *(_QWORD *)(*(_QWORD *)(v2 + 56) + 16 * (v3 & 0x7FFFFFFF) + 8);
  else
    v4 = *(_QWORD *)(*(_QWORD *)(v2 + 304) + 8 * v3);
  while ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 16);
    if ( *(_WORD *)(v5 + 68) == 32 )
    {
      v6 = sub_2E88FE0(*(_QWORD *)(v4 + 16)) + *(unsigned __int8 *)(*(_QWORD *)(v5 + 16) + 9LL);
      v7 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 40LL * (unsigned int)(v6 + 2) + 24) + v6 + 4;
      if ( v7 <= (unsigned int)sub_2EAB0A0(v4) )
        return 1;
    }
    v4 = *(_QWORD *)(v4 + 32);
  }
  return 0;
}
