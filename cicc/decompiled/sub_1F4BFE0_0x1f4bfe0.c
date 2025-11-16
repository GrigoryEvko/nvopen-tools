// Function: sub_1F4BFE0
// Address: 0x1f4bfe0
//
__int64 __fastcall sub_1F4BFE0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  _WORD *v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rcx
  unsigned __int16 *v18; // rax
  unsigned __int16 *v19; // rcx

  if ( *(_DWORD *)(a1 + 4) <= 1u )
    return 1;
  v6 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * a3 + 8);
  v7 = sub_1E15F70(a2);
  v9 = 0;
  v10 = *(_QWORD *)(v7 + 16);
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 112LL);
  if ( v11 != sub_1D00B10 )
    v9 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v11)(v10, a2, v8, 0);
  if ( (unsigned int)sub_1E165A0(a4, v6, 0, v9) == -1 )
  {
    v12 = *(_QWORD *)(a1 + 184);
    v13 = *(__int64 (**)())(*(_QWORD *)v12 + 656LL);
    if ( v13 != sub_1D918C0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v13)(v12, a4) )
        return sub_1F4BF20(a1, a2, 1);
    }
  }
  if ( !sub_1F4B670(a1) )
    return 0;
  v15 = sub_1F4B8B0(a1, a2);
  if ( (*v15 & 0x3FFF) == 0x3FFF )
    return 0;
  v16 = (unsigned __int16)v15[1];
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 176) + 136LL);
  v18 = (unsigned __int16 *)(v17 + 4 * v16);
  v19 = (unsigned __int16 *)(v17 + 4 * (v16 + (unsigned __int16)v15[2]));
  if ( v19 == v18 )
    return 0;
  while ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 32LL * *v18 + 16) )
  {
    v18 += 2;
    if ( v18 == v19 )
      return 0;
  }
  return 1;
}
