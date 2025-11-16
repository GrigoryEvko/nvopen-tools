// Function: sub_2AAE650
// Address: 0x2aae650
//
__int64 __fastcall sub_2AAE650(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  char v8; // r8

  if ( (*(_BYTE *)(*(_QWORD *)a1 + 4LL) || **(_DWORD **)a1 != 1)
    && (unsigned __int8)sub_B19060(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 416LL) + 56LL, *(_QWORD *)(a2 + 40), a3, a4) )
  {
    v5 = *(_QWORD *)(a1 + 8);
    switch ( (unsigned int)sub_2AAA2B0(v5, a2, **(_DWORD **)a1, *(_BYTE *)(*(_QWORD *)a1 + 4LL)) )
    {
      case 1u:
      case 5u:
        v8 = sub_B19060(*(_QWORD *)(v5 + 440) + 440LL, a2, v6, v7);
        result = 2;
        if ( v8 )
          return result;
        return 1;
      case 2u:
        return 5;
      case 3u:
        return 4;
      case 4u:
        return 3;
      case 6u:
      case 7u:
        sub_C64FA0("Instr has invalid widening decision", 0, 0);
      default:
        BUG();
    }
  }
  return 1;
}
