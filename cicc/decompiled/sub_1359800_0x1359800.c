// Function: sub_1359800
// Address: 0x1359800
//
void *__fastcall sub_1359800(_QWORD *a1, __int64 a2, __int64 a3)
{
  a1[1] = 2;
  a1[2] = 0;
  a1[3] = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220(a1 + 1);
  a1[4] = a3;
  *a1 = &unk_49E85F8;
  return &unk_49E85F8;
}
