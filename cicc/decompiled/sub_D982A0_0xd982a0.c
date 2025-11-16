// Function: sub_D982A0
// Address: 0xd982a0
//
void *__fastcall sub_D982A0(_QWORD *a1, __int64 a2, __int64 a3)
{
  a1[1] = 2;
  a1[2] = 0;
  a1[3] = a2;
  if ( a2 != -4096 && a2 != 0 && a2 != -8192 )
    sub_BD73F0((__int64)(a1 + 1));
  a1[4] = a3;
  *a1 = &unk_49DE910;
  return &unk_49DE910;
}
