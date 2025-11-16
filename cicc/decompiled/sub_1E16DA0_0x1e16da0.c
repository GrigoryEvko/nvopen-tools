// Function: sub_1E16DA0
// Address: 0x1e16da0
//
__int64 __fastcall sub_1E16DA0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14)
{
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  _BYTE *v20; // r9
  _WORD *v21; // rsi
  __int64 v22; // r14
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rax
  int v26; // edx
  unsigned int v27; // eax

  v17 = sub_1E15F70(a1);
  v21 = *(_WORD **)(a1 + 16);
  v22 = v17;
  if ( *v21 != 1 )
    return sub_1F3AD60(a3, v21, a2, a4, v17, v20, a7, a8, a9, a10, a11, a12, a13, a14);
  v23 = *(_QWORD *)(a1 + 32) + 40LL * a2;
  if ( *(_BYTE *)v23 )
    return 0;
  if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 && (*(_WORD *)(v23 + 2) & 0xFF0) != 0 )
    a2 = sub_1E16AB0(a1, a2, 5LL * a2, v18, v19, v20);
  v24 = sub_1E16480(a1, a2, 0);
  if ( v24 < 0 )
    return 0;
  v25 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL * v24 + 24);
  v26 = v25 & 7;
  if ( (unsigned int)(v26 - 1) <= 2 )
  {
    if ( (int)v25 >= 0 )
    {
      v27 = WORD1(v25);
      if ( v27 )
        return *(_QWORD *)(a4[32] + 8LL * (v27 - 1));
    }
    return 0;
  }
  if ( v26 != 6 )
    return 0;
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a4 + 144LL))(a4, v22, 0);
}
