// Function: sub_28E8D40
// Address: 0x28e8d40
//
__int64 __fastcall sub_28E8D40(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  const char *v4; // rax
  __int64 v5; // rdx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax

  if ( sub_B2FC80(a2) )
    return 0;
  if ( (*(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFF8LL) == a2 + 72 )
    return 0;
  v4 = sub_BD5D20(a2);
  if ( v5 == 17 && !(*(_QWORD *)v4 ^ 0x70656661732E6367LL | *((_QWORD *)v4 + 1) ^ 0x6C6F705F746E696FLL) && v4[16] == 108 )
    return 0;
  if ( (*(_BYTE *)(a2 + 3) & 0x40) == 0 )
    return 0;
  v7 = (__int64 *)sub_B2DBE0(a2);
  v11 = *v7;
  v12 = v7[1];
  if ( v12 == 18 )
  {
    v8 = *(_QWORD *)v11 ^ 0x696F706574617473LL | *(_QWORD *)(v11 + 8) ^ 0x706D6178652D746ELL;
    if ( !v8 && *(_WORD *)(v11 + 16) == 25964 )
      return sub_28E78A0(a2, a3, v11, v8, v9, v10);
    return 0;
  }
  if ( v12 != 7 || *(_DWORD *)v11 != 1701998435 || *(_WORD *)(v11 + 4) != 27747 || *(_BYTE *)(v11 + 6) != 114 )
    return 0;
  return sub_28E78A0(a2, a3, v11, v8, v9, v10);
}
