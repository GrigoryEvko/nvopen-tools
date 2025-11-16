// Function: sub_278B7A0
// Address: 0x278b7a0
//
unsigned __int8 *__fastcall sub_278B7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rsi
  unsigned __int64 v8; // rdi
  int v9; // eax
  unsigned __int8 *v10; // r12
  unsigned __int8 *v11; // rdi
  int v12; // eax
  unsigned __int8 *v13; // r15
  __int64 v14; // r15
  __int64 v15; // r12
  int v16; // r13d
  unsigned __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  v4 = (_QWORD *)(a2 + 48);
  v8 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v8 == v4 )
  {
    v10 = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = 0;
    v11 = (unsigned __int8 *)(v8 - 24);
    if ( (unsigned int)(v9 - 30) < 0xB )
      v10 = v11;
  }
  if ( (unsigned int)sub_B46E30((__int64)v10) != 2 )
    return 0;
  v12 = *v10;
  if ( (unsigned int)(v12 - 29) > 6 )
  {
    v13 = 0;
    if ( (unsigned int)(v12 - 37) <= 3 )
      return v13;
  }
  else
  {
    v13 = 0;
    if ( (unsigned int)(v12 - 29) > 4 )
      return v13;
  }
  v14 = sub_B46EC0((__int64)v10, 0);
  if ( v14 == a3 )
    v14 = sub_B46EC0((__int64)v10, 1u);
  if ( !sub_AA54C0(v14) )
    return 0;
  v15 = *(_QWORD *)(v14 + 56);
  v16 = qword_4FFB888;
  v19 = v14 + 48;
  if ( v14 + 48 == v15 )
    return 0;
  while ( 1 )
  {
    v13 = 0;
    if ( v15 )
      v13 = (unsigned __int8 *)(v15 - 24);
    if ( !sub_B46AA0((__int64)v13) )
    {
      if ( !--v16 )
        return 0;
      if ( sub_B46220((__int64)v13, a4) )
        break;
    }
    v15 = *(_QWORD *)(v15 + 8);
    if ( v19 == v15 )
      return 0;
  }
  v17 = sub_1037A30(*(_QWORD *)(a1 + 16), v13, 1);
  if ( (v17 & 7) != 3 || v17 >> 61 != 1 || (unsigned __int8)sub_30ED170(*(_QWORD *)(a1 + 104), v13) )
    return 0;
  return v13;
}
