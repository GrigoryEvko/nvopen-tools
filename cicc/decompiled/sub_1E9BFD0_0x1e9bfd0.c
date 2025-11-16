// Function: sub_1E9BFD0
// Address: 0x1e9bfd0
//
__int64 __fastcall sub_1E9BFD0(__int64 a1, int a2, __int16 a3)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  unsigned int *v7; // r12

  v3 = *(unsigned int *)(a1 + 16);
  if ( (v3 & 1) == 0 )
    return 0;
  v5 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)v3 > *(_DWORD *)(v5 + 40) )
    return 0;
  v7 = (unsigned int *)(*(_QWORD *)(v5 + 32) + 40 * v3);
  sub_1E310D0((__int64)v7, a2);
  *v7 = *v7 & 0xFFF000FF | ((a3 & 0xFFF) << 8);
  return 1;
}
