// Function: sub_16D42A0
// Address: 0x16d42a0
//
__int64 sub_16D42A0()
{
  char ***v0; // rax
  char *v1; // rcx
  char *v2; // rdi

  v0 = *(char ****)(__readfsqword(0) - 24);
  v1 = **v0;
  v2 = &(*v0)[1][(_QWORD)*v0[1]];
  if ( ((unsigned __int8)v1 & 1) != 0 )
    v1 = *(char **)&v1[*(_QWORD *)v2 - 1];
  return ((__int64 (__fastcall *)(char *, _QWORD, _QWORD))v1)(v2, *v0[2], *v0[3]);
}
