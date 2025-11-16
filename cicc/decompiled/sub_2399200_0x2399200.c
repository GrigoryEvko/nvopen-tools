// Function: sub_2399200
// Address: 0x2399200
//
void *__fastcall sub_2399200(_QWORD *a1)
{
  __int64 v1; // rbx
  void *result; // rax

  v1 = a1[1];
  result = &unk_4A0AB10;
  *a1 = &unk_4A0AB10;
  if ( v1 )
  {
    sub_2398D90(v1 + 64);
    return (void *)sub_2398F30(v1 + 32);
  }
  return result;
}
