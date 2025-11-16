// Function: sub_2C14A00
// Address: 0x2c14a00
//
__int64 __fastcall sub_2C14A00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  bool v9; // of
  unsigned __int64 v10; // rbx

  if ( !*(_BYTE *)(a1 + 104) || *(_BYTE *)(a1 + 106) )
    return sub_2C14770(a1, a2, a3);
  v4 = *(_QWORD *)(a1 + 96);
  if ( *(_BYTE *)v4 != 61 )
    v4 = *(_QWORD *)(v4 - 64);
  v5 = sub_2AAEDF0(*(_QWORD *)(v4 + 8), a2);
  sub_2AAE0E0(*(_QWORD *)(a1 + 96));
  result = sub_DFD500(*(_QWORD *)a3);
  v7 = result;
  if ( *(_BYTE *)(a1 + 105) )
  {
    v8 = sub_DFBC30(*(__int64 **)a3, 1, v5, 0, 0, *(unsigned int *)(a3 + 176), 0, 0, 0, 0, 0);
    v9 = __OFADD__(v8, v7);
    v10 = v8 + v7;
    if ( v9 )
    {
      v10 = 0x8000000000000000LL;
      if ( v8 > 0 )
        return 0x7FFFFFFFFFFFFFFFLL;
    }
    return v10;
  }
  return result;
}
