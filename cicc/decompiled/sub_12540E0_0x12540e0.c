// Function: sub_12540E0
// Address: 0x12540e0
//
unsigned __int64 *__fastcall sub_12540E0(unsigned __int64 *a1, _QWORD *a2, unsigned __int64 a3, _QWORD *a4)
{
  unsigned __int64 v7; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 > (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) )
  {
    sub_1253E40(v8, 3u);
  }
  else
  {
    if ( (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) >= a3 + 1 )
      goto LABEL_6;
    sub_1253E40(v8, 1u);
  }
  if ( (v8[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
LABEL_6:
  v7 = a2[3] - a3;
  *a4 = a2[2] + a3;
  a4[1] = v7;
  *a1 = 1;
  return a1;
}
