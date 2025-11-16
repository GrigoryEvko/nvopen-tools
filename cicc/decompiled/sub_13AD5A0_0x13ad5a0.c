// Function: sub_13AD5A0
// Address: 0x13ad5a0
//
__int64 __fastcall sub_13AD5A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned int v9; // [rsp+Ch] [rbp-34h]

  if ( *(_WORD *)(a2 + 24) != 7 )
    return a2;
  v3 = *(_QWORD *)(a2 + 48);
  if ( a3 == v3 )
    return **(_QWORD **)(a2 + 32);
  v4 = *(_QWORD *)(a1 + 8);
  v9 = *(_WORD *)(a2 + 26) & 7;
  v5 = sub_13A5BC0((_QWORD *)a2, v4);
  v6 = sub_13AD5A0(a1, **(_QWORD **)(a2 + 32), a3);
  return sub_14799E0(v4, v6, v5, v3, v9);
}
