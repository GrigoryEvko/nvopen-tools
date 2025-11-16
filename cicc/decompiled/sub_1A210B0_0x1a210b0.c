// Function: sub_1A210B0
// Address: 0x1a210b0
//
__int64 __fastcall sub_1A210B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r15
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v3 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned __int8)v3 <= 0x10u )
    {
      v4 = 100990;
      if ( _bittest64(&v4, v3) )
        break;
    }
    v6 = (unsigned int)sub_15A9FE0(a1, a2);
    v7 = (v6 + ((unsigned __int64)(sub_127FA20(a1, a2) + 7) >> 3) - 1) / v6 * v6;
    v8 = sub_127FA20(a1, a2);
    v9 = *(_BYTE *)(a2 + 8);
    if ( v9 == 14 )
    {
      v10 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      if ( v9 != 13 )
        return a2;
      v11 = sub_15A9930(a1, a2);
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (unsigned int)sub_15A8020(v11, 0));
    }
    v12 = (unsigned int)sub_15A9FE0(a1, v10);
    if ( v7 > v12 * ((v12 + ((unsigned __int64)(sub_127FA20(a1, v10) + 7) >> 3) - 1) / v12) || sub_127FA20(a1, v10) < v8 )
      break;
    a2 = v10;
  }
  return a2;
}
