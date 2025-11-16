// Function: sub_1952070
// Address: 0x1952070
//
void __fastcall sub_1952070(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // r14
  _QWORD *i; // rbx
  __int64 v5; // r15

  v2 = a1[5];
  sub_1AEC3A0();
  v3 = (_QWORD *)(v2 + 40);
  for ( i = (_QWORD *)(*(_QWORD *)(v2 + 40) & 0xFFFFFFFFFFFFFFF8LL); v3 != i; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( i )
    {
      v5 = (__int64)(i - 3);
      if ( a1 == i - 3 || !(unsigned __int8)sub_14AE440((__int64)(i - 3)) )
        break;
    }
    else
    {
      v5 = 0;
      if ( !(unsigned __int8)sub_14AE440(0) )
        break;
    }
    sub_1648780(v5, (__int64)a1, a2);
  }
  if ( !a1[1] && !(unsigned __int8)sub_15F3040((__int64)a1) && !sub_15F3330((__int64)a1) )
    sub_15F20C0(a1);
}
