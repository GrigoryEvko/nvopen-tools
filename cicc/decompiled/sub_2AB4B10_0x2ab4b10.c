// Function: sub_2AB4B10
// Address: 0x2ab4b10
//
unsigned __int64 __fastcall sub_2AB4B10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // r15
  unsigned __int8 **v9; // rax
  __int64 v10; // rcx
  bool v11; // of
  int v12; // [rsp+4h] [rbp-3Ch]

  if ( *(_BYTE *)a2 == 61 )
    v3 = *(_QWORD *)(a2 + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v4 = sub_2AAEDF0(v3, a3);
  v5 = sub_228AED0((_BYTE *)a2);
  v12 = sub_31A5150(*(_QWORD *)(a1 + 440), v3, v5);
  if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, a2, v6, 63) )
  {
    v7 = sub_DFD500(*(_QWORD *)(a1 + 448));
    if ( v12 >= 0 )
      return v7;
  }
  else
  {
    v9 = (unsigned __int8 **)sub_986520(a2);
    sub_DFB770(*v9);
    v7 = sub_DFD4A0(*(__int64 **)(a1 + 448));
    if ( v12 >= 0 )
      return v7;
  }
  v10 = sub_DFBC30(*(__int64 **)(a1 + 448), 1, v4, 0, 0, *(unsigned int *)(a1 + 992), 0, 0, 0, 0, 0);
  v11 = __OFADD__(v10, v7);
  v7 += v10;
  if ( v11 )
  {
    v7 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v10 <= 0 )
      return 0x8000000000000000LL;
  }
  return v7;
}
