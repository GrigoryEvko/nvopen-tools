// Function: sub_2260650
// Address: 0x2260650
//
void *__fastcall sub_2260650(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  void (***v3)(void); // rdi
  void (*v4)(void); // rdx
  __int64 (***v5)(void); // rdi
  void *result; // rax
  __int64 (*v7)(void); // rdx

  a1[2] = &unk_4A08390;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[9];
  if ( v2 )
    v2(a1 + 7, a1 + 7, 3);
  v3 = (void (***)(void))a1[6];
  if ( v3 )
  {
    v4 = **v3;
    if ( (char *)v4 == (char *)sub_16D4120 )
      (*v3)[2]();
    else
      v4();
  }
  v5 = (__int64 (***)(void))a1[3];
  result = &unk_49EF4E8;
  a1[2] = &unk_49EF4E8;
  if ( v5 )
  {
    v7 = **v5;
    if ( (char *)v7 == (char *)sub_16D4120 )
      return (void *)(*v5)[2]();
    else
      return (void *)v7();
  }
  return result;
}
