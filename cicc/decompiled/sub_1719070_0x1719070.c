// Function: sub_1719070
// Address: 0x1719070
//
__int64 __fastcall sub_1719070(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r15
  __int16 v7; // bx
  void *v8; // rax
  void *v9; // r14
  _QWORD *v10; // rdi
  __int64 v12; // rdx

  v7 = *(_WORD *)(a1 + 2);
  v8 = sub_16982C0();
  v9 = v8;
  if ( v7 <= 0 )
  {
    v12 = -v7;
    if ( (void *)a2 == v8 )
      sub_169C630((_QWORD *)(a1 + 16), (__int64)v8, v12);
    else
      sub_1699170(a1 + 16, a2, v12);
    v5 = a1 + 16;
    if ( *(void **)(a1 + 16) == v9 )
      sub_169C8D0(v5, a3, a4, a5);
    else
      sub_1699490(v5);
  }
  else
  {
    v10 = (_QWORD *)(a1 + 16);
    if ( (void *)a2 == v8 )
      sub_169C630(v10, a2, v7);
    else
      sub_1699170((__int64)v10, a2, v7);
  }
  *(_WORD *)a1 = 257;
  return 257;
}
