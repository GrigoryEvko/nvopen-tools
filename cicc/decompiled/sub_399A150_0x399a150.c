// Function: sub_399A150
// Address: 0x399a150
//
__int64 __fastcall sub_399A150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 result; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  char v14; // al

  v5 = *(_QWORD *)(a3 + 8);
  if ( !*(_BYTE *)(a1 + 4513) )
  {
    v6 = *(unsigned int *)(v5 + 8);
LABEL_3:
    v7 = *(_QWORD *)(v5 + 8 * (5 - v6));
    goto LABEL_4;
  }
  v14 = ((__int64 (*)(void))sub_3989C80)();
  v6 = *(unsigned int *)(v5 + 8);
  if ( v14 )
    goto LABEL_3;
  v7 = *(_QWORD *)(v5 + 8 * (5 - v6));
  if ( !*(_BYTE *)(v7 + 48) )
    return sub_39CEC10(a2, a3);
LABEL_4:
  v8 = sub_3999410(a1, v7);
  v9 = *(_QWORD *)(v8 + 616);
  v10 = v8;
  if ( !v9 )
  {
    v12 = a3;
    v13 = v8;
    return sub_39CEC10(v13, v12);
  }
  if ( (unsigned __int8)sub_3989C80(a1) )
    a2 = v10;
  sub_39CEC10(a2, a3);
  result = *(_QWORD *)(v10 + 80);
  v12 = a3;
  v13 = v9;
  if ( *(_BYTE *)(result + 48) )
    return sub_39CEC10(v13, v12);
  return result;
}
