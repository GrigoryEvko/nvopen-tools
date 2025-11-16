// Function: sub_169C8D0
// Address: 0x169c8d0
//
__int64 __fastcall sub_169C8D0(__int64 a1, double a2, double a3, double a4)
{
  void *v5; // rbx
  void **v6; // rdi
  __int64 v7; // rax

  v5 = sub_16982C0();
  do
  {
    v6 = (void **)(*(_QWORD *)(a1 + 8) + 8LL);
    if ( *v6 == v5 )
      sub_169C8D0(v6, a2, a3, a4);
    else
      sub_1699490((__int64)v6);
    v7 = *(_QWORD *)(a1 + 8);
    a1 = v7 + 40;
  }
  while ( *(void **)(v7 + 40) == v5 );
  return sub_1699490(v7 + 40);
}
