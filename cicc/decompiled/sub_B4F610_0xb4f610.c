// Function: sub_B4F610
// Address: 0xb4f610
//
__int64 __fastcall sub_B4F610(__int64 a1)
{
  unsigned __int8 *v1; // rdx
  __int64 v3; // rax
  int v4; // edx

  v1 = *(unsigned __int8 **)(a1 - 64);
  if ( (unsigned int)*v1 - 12 <= 1 )
    return 0;
  if ( (unsigned int)**(unsigned __int8 **)(a1 - 32) - 12 <= 1 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v3 + 8) == 18 )
    return 0;
  v4 = 2 * *(_DWORD *)(*((_QWORD *)v1 + 1) + 32LL);
  if ( v4 != *(_DWORD *)(v3 + 32) )
    return 0;
  else
    return sub_B487F0(*(int **)(a1 + 72), *(unsigned int *)(a1 + 80), v4);
}
