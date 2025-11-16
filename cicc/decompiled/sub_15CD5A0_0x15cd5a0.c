// Function: sub_15CD5A0
// Address: 0x15cd5a0
//
_QWORD *__fastcall sub_15CD5A0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // rbx
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rdi

  v3 = a2 + 40;
  v4 = *(_QWORD *)(a2 + 40);
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 40 != (v4 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    do
    {
      if ( !v5 )
        BUG();
      if ( *(_QWORD *)(v5 - 16) )
      {
        v6 = sub_1599EF0(*(__int64 ***)(v5 - 24));
        sub_164D160(v5 - 24, v6);
        v4 = *(_QWORD *)(a2 + 40);
      }
      v7 = (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
      sub_157EA20(v3, (__int64)(v7 - 3));
      v8 = (unsigned __int64 *)v7[1];
      v9 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      *v8 = v9 | *v8 & 7;
      *(_QWORD *)(v9 + 8) = v8;
      *v7 &= 7uLL;
      v7[1] = 0;
      sub_164BEC0(v7 - 3, v7 - 3, v9, v8, v10);
      v4 = *(_QWORD *)(a2 + 40);
      v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    }
    while ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != v3 );
  }
  v11 = sub_157E9C0(a2);
  v12 = sub_1648A60(56, 0);
  if ( v12 )
    sub_15F82E0(v12, v11, a2);
  return sub_1412190(a1 + 280, a2);
}
