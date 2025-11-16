// Function: sub_D68A80
// Address: 0xd68a80
//
__int64 __fastcall sub_D68A80(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax

  v2 = 32LL * ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) - 1);
  v3 = 8LL * ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) - 1);
  sub_AC2B30(32LL * a2 + *(_QWORD *)(a1 - 8), *(_QWORD *)(*(_QWORD *)(a1 - 8) + v2));
  *(_QWORD *)(*(_QWORD *)(a1 - 8) + 8LL * a2 + 32LL * *(unsigned int *)(a1 + 76)) = *(_QWORD *)(*(_QWORD *)(a1 - 8)
                                                                                              + v3
                                                                                              + 32LL
                                                                                              * *(unsigned int *)(a1 + 76));
  v4 = v2 + *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    **(_QWORD **)(v4 + 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v4 + 16);
  }
  *(_QWORD *)v4 = 0;
  *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL * *(unsigned int *)(a1 + 76) + v3) = 0;
  result = (*(_DWORD *)(a1 + 4) + 0x7FFFFFF) & 0x7FFFFFF | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  return result;
}
