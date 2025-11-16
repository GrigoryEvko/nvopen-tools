// Function: sub_2AB4CE0
// Address: 0x2ab4ce0
//
unsigned __int64 __fastcall sub_2AB4CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  bool v8; // of
  unsigned __int64 v9; // rax
  unsigned __int64 result; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rax

  if ( *(_BYTE *)a2 == 61 )
    v3 = *(_QWORD *)(a2 + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v4 = sub_2AAEDF0(v3, a3);
  if ( *(_BYTE *)a2 != 61 )
  {
    v5 = 0;
    if ( !(unsigned __int8)sub_31A5290(*(_QWORD *)(a1 + 440), *(_QWORD *)(a2 - 64)) )
      v5 = sub_DFD330(*(__int64 **)(a1 + 448));
    v6 = sub_DFD4A0(*(__int64 **)(a1 + 448));
    v7 = sub_DFDB90(*(_QWORD *)(a1 + 448));
    v8 = __OFADD__(v6, v7);
    v9 = v6 + v7;
    if ( v8 )
    {
      v9 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v6 <= 0 )
        v9 = 0x8000000000000000LL;
    }
    v8 = __OFADD__(v5, v9);
    result = v5 + v9;
    if ( !v8 )
      return result;
LABEL_11:
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 <= 0 )
      return 0x8000000000000000LL;
    return result;
  }
  v5 = sub_DFBC30(*(__int64 **)(a1 + 448), 0, v4, 0, 0, *(unsigned int *)(a1 + 992), 0, 0, 0, 0, 0);
  v11 = sub_DFD4A0(*(__int64 **)(a1 + 448));
  v12 = sub_DFDB90(*(_QWORD *)(a1 + 448));
  v8 = __OFADD__(v11, v12);
  v13 = v11 + v12;
  if ( v8 )
  {
    v13 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v11 <= 0 )
      v13 = 0x8000000000000000LL;
  }
  v8 = __OFADD__(v5, v13);
  result = v5 + v13;
  if ( v8 )
    goto LABEL_11;
  return result;
}
