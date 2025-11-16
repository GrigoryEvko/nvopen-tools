// Function: sub_1D18440
// Address: 0x1d18440
//
__int64 __fastcall sub_1D18440(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 (*v7)(); // rax
  char v8; // si
  unsigned int *v9; // rdx
  unsigned int *i; // rdi
  __int64 result; // rax
  __int64 j; // r12

  v4 = (__int64 *)a1[2];
  v5 = *v4;
  v6 = *(__int64 (**)(void))(*v4 + 992);
  if ( v6 != sub_1D12E40 )
  {
    result = v6();
    if ( (_BYTE)result )
      return result;
    v4 = (__int64 *)a1[2];
    v5 = *v4;
  }
  v7 = *(__int64 (**)())(v5 + 984);
  v8 = 0;
  if ( v7 != sub_1D12E30 )
    v8 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, _QWORD))v7)(v4, a2, a1[9], a1[8]);
  v9 = *(unsigned int **)(a2 + 32);
  for ( i = &v9[10 * *(unsigned int *)(a2 + 56)]; i != v9; v9 += 10 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v9 + 40LL) + 16LL * v9[2]) != 1 )
      v8 |= (*(_BYTE *)(*(_QWORD *)v9 + 26LL) & 4) != 0;
  }
  result = *(unsigned __int8 *)(a2 + 26);
  if ( ((*(_BYTE *)(a2 + 26) & 4) != 0) != v8 )
  {
    result = (unsigned int)result & 0xFFFFFFFB;
    *(_BYTE *)(a2 + 26) = result | (4 * (v8 & 1));
    for ( j = *(_QWORD *)(a2 + 48); j; j = *(_QWORD *)(j + 32) )
      result = sub_1D18440(a1, *(_QWORD *)(j + 16));
  }
  return result;
}
