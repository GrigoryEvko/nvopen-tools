// Function: sub_C405F0
// Address: 0xc405f0
//
__int64 __fastcall sub_C405F0(__int64 a1)
{
  void *v2; // rbx
  void **v3; // rdi
  char v4; // al
  __int64 v6; // rax

  v2 = sub_C33340();
  do
  {
    v3 = *(void ***)(a1 + 8);
    if ( *v3 == v2 )
      v4 = sub_C405F0();
    else
      v4 = sub_C3BCA0((__int64)v3);
    if ( !v4 )
      return 0;
    v6 = *(_QWORD *)(a1 + 8);
    a1 = v6 + 24;
  }
  while ( v2 == *(void **)(v6 + 24) );
  return sub_C3BCA0(v6 + 24);
}
