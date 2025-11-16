// Function: sub_15C7110
// Address: 0x15c7110
//
_QWORD *__fastcall sub_15C7110(_QWORD *a1, int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rdi
  __int64 v7; // rax

  if ( a4 )
  {
    v6 = (__int64 *)(*(_QWORD *)(a4 + 16) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a4 + 16) & 4) != 0 )
      v6 = (__int64 *)*v6;
    v7 = sub_15B9E00(v6, a2, a3, a4, a5, 0, 1);
    sub_15C7080(a1, v7);
    return a1;
  }
  else
  {
    *a1 = 0;
    return a1;
  }
}
