// Function: sub_1B92200
// Address: 0x1b92200
//
bool __fastcall sub_1B92200(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 **v3; // r15
  __int64 v4; // r14
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 *v11; // r14
  _QWORD *v12; // r15
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r13
  __int64 v16; // rbx

  v3 = (__int64 **)a2;
  v4 = 0;
  v5 = a3;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 != 54 )
  {
    v3 = 0;
    if ( v6 == 55 )
      v4 = a2;
  }
  v7 = sub_13A4950(a2);
  if ( !(unsigned int)sub_1BF20B0(*(_QWORD *)(a1 + 320), v7) || (unsigned __int8)sub_1B91FD0(a1, a2) )
    return 0;
  v9 = sub_15F2050(a2);
  v10 = sub_1632FA0(v9);
  if ( v3 )
    v11 = *v3;
  else
    v11 = **(__int64 ***)(v4 - 48);
  if ( (unsigned int)v5 <= 1 )
  {
    v15 = (unsigned int)sub_15A9FE0(v10, (__int64)v11);
    v16 = sub_127FA20(v10, (__int64)v11);
    return 8 * v15 * ((v15 + ((unsigned __int64)(v16 + 7) >> 3) - 1) / v15) == sub_127FA20(v10, (__int64)v11);
  }
  else
  {
    v12 = sub_16463B0(v11, v5);
    v13 = (unsigned int)sub_15A9FE0(v10, (__int64)v11);
    v14 = (v13 + ((unsigned __int64)(sub_127FA20(v10, (__int64)v11) + 7) >> 3) - 1) / v13 * v13 * v5;
    return (unsigned __int64)(sub_127FA20(v10, (__int64)v12) + 7) >> 3 == v14;
  }
}
