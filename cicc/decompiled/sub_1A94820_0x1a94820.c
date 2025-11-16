// Function: sub_1A94820
// Address: 0x1a94820
//
__int64 __fastcall sub_1A94820(__int64 a1, unsigned __int64 a2)
{
  const char *v2; // rax
  __int64 v3; // rdx
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  if ( sub_15E4F60(a2) )
    return 0;
  if ( a2 + 72 == (*(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v2 = sub_1649960(a2);
  if ( v3 == 17 && !(*(_QWORD *)v2 ^ 0x70656661732E6367LL | *((_QWORD *)v2 + 1) ^ 0x6C6F705F746E696FLL) && v2[16] == 108 )
    return 0;
  if ( (*(_BYTE *)(a2 + 19) & 0x40) == 0 )
    return 0;
  v5 = (__int64 *)sub_15E0FA0(a2);
  v6 = *v5;
  v7 = v5[1];
  if ( v7 == 18 )
  {
    if ( !(*(_QWORD *)v6 ^ 0x696F706574617473LL | *(_QWORD *)(v6 + 8) ^ 0x706D6178652D746ELL)
      && *(_WORD *)(v6 + 16) == 25964 )
    {
      return sub_1A93470(a1, a2);
    }
    return 0;
  }
  if ( v7 != 7 || *(_DWORD *)v6 != 1701998435 || *(_WORD *)(v6 + 4) != 27747 || *(_BYTE *)(v6 + 6) != 114 )
    return 0;
  return sub_1A93470(a1, a2);
}
