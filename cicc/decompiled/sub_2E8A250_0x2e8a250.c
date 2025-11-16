// Function: sub_2E8A250
// Address: 0x2e8a250
//
__int64 __fastcall sub_2E8A250(__int64 a1, unsigned int a2, __int64 a3, _QWORD *a4)
{
  unsigned int v6; // r12d
  __int64 v7; // r15
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rax
  int v11; // edx
  unsigned int v12; // eax

  v6 = a2;
  v7 = sub_2E88D60(a1);
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 )
    return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD *, __int64))(*(_QWORD *)a3 + 16LL))(
             a3,
             *(_QWORD *)(a1 + 16),
             a2,
             a4,
             v7);
  v8 = *(_QWORD *)(a1 + 32) + 40LL * a2;
  if ( *(_BYTE *)v8 )
    return 0;
  if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 && (*(_WORD *)(v8 + 2) & 0xFF0) != 0 )
    v6 = sub_2E89F40(a1, a2);
  v9 = sub_2E890A0(a1, v6, 0);
  if ( v9 < 0 )
    return 0;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL * v9 + 24);
  v11 = v10 & 7;
  if ( ((unsigned __int8)(v11 - 2) <= 1u || v11 == 1) && (int)v10 >= 0 )
  {
    v12 = WORD1(v10) & 0x3FFF;
    if ( v12 )
      return *(_QWORD *)(a4[35] + 8LL * (v12 - 1));
  }
  if ( v11 != 6 )
    return 0;
  else
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a4 + 336LL))(a4, v7, 0);
}
