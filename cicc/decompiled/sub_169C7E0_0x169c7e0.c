// Function: sub_169C7E0
// Address: 0x169c7e0
//
void *__fastcall sub_169C7E0(_QWORD *a1, _QWORD *a2)
{
  *a1 = *a2;
  a1[1] = a2[1];
  a2[1] = 0;
  *a2 = &unk_42AE9A0;
  return &unk_42AE9A0;
}
