// Function: sub_1457D90
// Address: 0x1457d90
//
void *__fastcall sub_1457D90(_QWORD *a1, __int64 a2, __int64 a3)
{
  a1[1] = 2;
  a1[2] = 0;
  a1[3] = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220(a1 + 1);
  a1[4] = a3;
  *a1 = &unk_49EC5C8;
  return &unk_49EC5C8;
}
