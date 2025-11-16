// Function: sub_2F28D40
// Address: 0x2f28d40
//
__int64 __fastcall sub_2F28D40(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v3; // r8d
  __int64 v4; // rax
  unsigned int *v5; // rcx
  unsigned int v6; // eax
  unsigned int *v7; // rcx
  unsigned int v8; // eax

  v3 = 0;
  v4 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
  *(_DWORD *)(a1 + 16) = v4;
  if ( (int)v4 <= 1 )
  {
    v3 = 1;
    v5 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40 * v4);
    v6 = *v5;
    *a2 = v5[2];
    a2[1] = (v6 >> 8) & 0xFFF;
    v7 = *(unsigned int **)(*(_QWORD *)(a1 + 8) + 32LL);
    v8 = *v7;
    *a3 = v7[2];
    a3[1] = (v8 >> 8) & 0xFFF;
  }
  return v3;
}
