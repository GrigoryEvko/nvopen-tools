// Function: sub_17578D0
// Address: 0x17578d0
//
__int64 __fastcall sub_17578D0(__int64 *a1, __int64 a2, char a3)
{
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  __int64 v10; // rsi
  _QWORD v11[2]; // [rsp+8h] [rbp-18h] BYREF

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 17 )
      return result;
    v5 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
    if ( v5 )
      v5 -= 24;
    result = sub_157EE30(v5);
    if ( !result )
      BUG();
LABEL_6:
    v6 = *(_QWORD *)(result + 16);
    a1[2] = result;
    a1[1] = v6;
    v7 = *(_QWORD *)(result + 24);
    v11[0] = v7;
    if ( !v7 )
      goto LABEL_7;
LABEL_12:
    result = sub_1623A60((__int64)v11, v7, 2);
    v8 = *a1;
    if ( !*a1 )
      goto LABEL_14;
    goto LABEL_13;
  }
  if ( (_BYTE)result == 77 )
  {
    result = sub_157EE30(*(_QWORD *)(a2 + 40));
    if ( !result )
      BUG();
    goto LABEL_6;
  }
  if ( !a3 )
  {
    v10 = *(_QWORD *)(a2 + 32);
    if ( !v10 )
      BUG();
    a2 = v10 - 24;
  }
  a1[1] = *(_QWORD *)(a2 + 40);
  result = a2 + 24;
  a1[2] = a2 + 24;
  v7 = *(_QWORD *)(a2 + 48);
  v11[0] = v7;
  if ( v7 )
    goto LABEL_12;
LABEL_7:
  v8 = *a1;
  if ( !*a1 )
    return result;
LABEL_13:
  result = sub_161E7C0((__int64)a1, v8);
LABEL_14:
  v9 = (unsigned __int8 *)v11[0];
  *a1 = v11[0];
  if ( v9 )
    return sub_1623210((__int64)v11, v9, (__int64)a1);
  return result;
}
