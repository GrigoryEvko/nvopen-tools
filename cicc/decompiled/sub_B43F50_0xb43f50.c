// Function: sub_B43F50
// Address: 0xb43f50
//
_QWORD *__fastcall sub_B43F50(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v6; // rsi
  __int64 v8; // rdi
  char v10; // [rsp+Ch] [rbp-24h]

  v6 = *(_QWORD *)(a2 + 64);
  if ( !v6 )
    return &qword_4F81430[1];
  v8 = *(_QWORD *)(a1 + 64);
  if ( !v8 )
  {
    v10 = a5;
    sub_AA4580(*(_QWORD *)(a1 + 40), a1);
    v8 = *(_QWORD *)(a1 + 64);
    v6 = *(_QWORD *)(a2 + 64);
    a5 = v10;
  }
  return (_QWORD *)sub_B14600(v8, v6, a3, a4, a5);
}
