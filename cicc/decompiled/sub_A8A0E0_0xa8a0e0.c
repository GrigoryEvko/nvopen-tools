// Function: sub_A8A0E0
// Address: 0xa8a0e0
//
__int64 __fastcall sub_A8A0E0(__int64 a1, int *a2, const void *a3, __int64 a4)
{
  unsigned int v4; // r12d
  size_t v7; // rdx
  int v9; // eax

  v4 = *(unsigned __int8 *)(a1 + 20);
  if ( (_BYTE)v4 )
    return 0;
  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 != a4 || v7 && memcmp(*(const void **)a1, a3, v7) )
    return v4;
  v9 = *a2;
  *(_BYTE *)(a1 + 20) = 1;
  *(_DWORD *)(a1 + 16) = v9;
  return 1;
}
