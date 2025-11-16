// Function: sub_13A7760
// Address: 0x13a7760
//
__int64 __fastcall sub_13A7760(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // rax
  __int16 v8; // ax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // [rsp+10h] [rbp-40h]
  __int64 v12; // [rsp+18h] [rbp-38h]

  v4 = a4;
  v5 = a3;
  if ( (unsigned int)(a2 - 32) <= 1 )
  {
    v8 = *(_WORD *)(a3 + 24);
    if ( v8 == 3 )
    {
      if ( *(_WORD *)(a4 + 24) != 3 )
        goto LABEL_2;
    }
    else if ( v8 != 2 || *(_WORD *)(a4 + 24) != 2 )
    {
      goto LABEL_2;
    }
    v9 = *(_QWORD *)(a4 + 32);
    v11 = *(_QWORD *)(a3 + 32);
    v12 = sub_1456040(v11);
    v10 = sub_1456040(v9);
    if ( v12 == v10 )
      v4 = v9;
    a2 = (unsigned int)a2;
    if ( v12 == v10 )
      v5 = v11;
  }
LABEL_2:
  result = sub_147A340(*(_QWORD *)(a1 + 8), a2, v5, v4);
  if ( !(_BYTE)result )
  {
    v7 = sub_14806B0(*(_QWORD *)(a1 + 8), v5, v4, 0, 0);
    switch ( (int)a2 )
    {
      case ' ':
        result = sub_14560B0(v7);
        break;
      case '!':
        result = sub_1477CE0(*(_QWORD *)(a1 + 8), v7);
        break;
      case '"':
      case '#':
      case '$':
      case '%':
      case ')':
        result = sub_1477A90(*(_QWORD *)(a1 + 8), v7);
        break;
      case '&':
        result = sub_1477C30(*(_QWORD *)(a1 + 8), v7);
        break;
      case '\'':
        result = sub_1477BC0(*(_QWORD *)(a1 + 8), v7);
        break;
      case '(':
        result = sub_1477B50(*(_QWORD *)(a1 + 8), v7);
        break;
    }
  }
  return result;
}
