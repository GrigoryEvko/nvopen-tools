// Function: sub_B4F5D0
// Address: 0xb4f5d0
//
__int64 __fastcall sub_B4F5D0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // edx

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v1 + 8) == 18 )
    return 0;
  v2 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL);
  if ( v2 <= *(_DWORD *)(v1 + 32) )
    return 0;
  else
    return sub_B487F0(*(int **)(a1 + 72), *(unsigned int *)(a1 + 80), v2);
}
