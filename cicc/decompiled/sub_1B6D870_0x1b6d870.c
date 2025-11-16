// Function: sub_1B6D870
// Address: 0x1b6d870
//
__int64 __fastcall sub_1B6D870(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v3 )
  {
    v4 = 0;
    goto LABEL_5;
  }
  v4 = v3 - 24;
  result = 0;
  if ( a3 != (_QWORD *)v4 )
  {
LABEL_5:
    sub_1580AC0(a3, v4);
    return 1;
  }
  return result;
}
