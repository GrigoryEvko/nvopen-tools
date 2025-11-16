// Function: sub_4264C5
// Address: 0x4264c5
//
void __fastcall __noreturn sub_4264C5(unsigned int a1)
{
  __int64 v1; // r12
  __int64 v2; // rbp
  _QWORD v3[2]; // [rsp+0h] [rbp-48h] BYREF
  char v4; // [rsp+10h] [rbp-38h] BYREF

  v1 = sub_2252770(32);
  v2 = sub_2241E50();
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*(_QWORD *)v2 + 32LL))(v3, v2, a1);
  sub_2223570(v1, v3);
  if ( (char *)v3[0] != &v4 )
    j___libc_free_0();
  *(_DWORD *)(v1 + 16) = a1;
  *(_QWORD *)(v1 + 24) = v2;
  *(_QWORD *)v1 = off_4A07678;
  sub_2253480(v1, &`typeinfo for'std::system_error, sub_2241C70);
}
