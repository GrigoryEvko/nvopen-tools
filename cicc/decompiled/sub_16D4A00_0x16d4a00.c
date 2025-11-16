// Function: sub_16D4A00
// Address: 0x16d4a00
//
__int64 __fastcall sub_16D4A00(_QWORD *a1)
{
  void (__fastcall *v2)(_QWORD *, _QWORD *, __int64); // rax
  void (***v3)(void); // rdi
  void (*v4)(void); // rdx
  void (***v5)(void); // rdi
  void (*v6)(void); // rdx

  *a1 = &unk_49EF588;
  v2 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[7];
  if ( v2 )
    v2(a1 + 5, a1 + 5, 3);
  v3 = (void (***)(void))a1[4];
  *a1 = &unk_49EF540;
  if ( v3 )
  {
    v4 = **v3;
    if ( (char *)v4 == (char *)sub_16D4120 )
      (*v3)[2]();
    else
      v4();
  }
  v5 = (void (***)(void))a1[1];
  *a1 = &unk_49EF4E8;
  if ( v5 )
  {
    v6 = **v5;
    if ( (char *)v6 == (char *)sub_16D4120 )
      (*v5)[2]();
    else
      v6();
  }
  return j_j___libc_free_0(a1, 72);
}
