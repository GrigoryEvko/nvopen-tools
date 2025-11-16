// Function: sub_1A73F70
// Address: 0x1a73f70
//
__int64 __fastcall sub_1A73F70(__int64 a1, char a2)
{
  unsigned __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r12

  v2 = **(_QWORD **)(a1 + 760) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (**(_QWORD **)(a1 + 760) & 4) == 0 )
  {
    sub_1A72E50(a1, **(_QWORD **)(a1 + 760) & 0xFFFFFFFFFFFFFFF8LL);
    if ( !a2 )
      return v2;
    v4 = sub_157EE30(v2);
    if ( v4 )
    {
      if ( v4 == v2 + 40 )
        return v2;
    }
  }
  v5 = sub_1A73B00(a1, v2);
  sub_1A73460(a1, *(__int64 **)(a1 + 760), v5, 1);
  *(_QWORD *)(a1 + 760) = sub_1444DB0(*(_QWORD **)(a1 + 200), v5);
  return v5;
}
