// Function: sub_81AB10
// Address: 0x81ab10
//
__int64 __fastcall sub_81AB10(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rsi
  _QWORD *v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rbx
  void *v8; // r14
  unsigned __int64 v9; // rbx
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD *)(a2 + 56);
  if ( ~v4 <= a1 )
    goto LABEL_12;
  v5 = (_QWORD *)qword_4F194E8;
  v6 = v4 + a1;
  if ( !qword_4F194E8 )
  {
LABEL_7:
    v9 = v4 + a1 / 0xA + a1 - v3;
    if ( v3 >= v9 )
      v9 = v3;
    v7 = v3 + v9;
    if ( v7 + 1 >= v6 )
    {
      v8 = (void *)sub_822C60(*(_QWORD *)(a2 + 64), v3 + 1, v7 + 1);
      goto LABEL_11;
    }
LABEL_12:
    sub_685240(0x6D9u);
  }
  while ( 1 )
  {
    v7 = v5[9];
    if ( v7 >= v6 )
      break;
    v5 = (_QWORD *)*v5;
    if ( !v5 )
      goto LABEL_7;
  }
  v8 = (void *)v5[8];
  v5[8] = *(_QWORD *)(a2 + 64);
  v5[9] = *(_QWORD *)(a2 + 72);
  memcpy(v8, *(const void **)(a2 + 64), *(_QWORD *)(a2 + 56));
LABEL_11:
  result = sub_81A600(*(_QWORD *)(a2 + 64), *(_QWORD *)(a2 + 64) + v3, (__int64)v8, 1);
  *(_QWORD *)(a2 + 64) = v8;
  *(_QWORD *)(a2 + 72) = v7;
  return result;
}
