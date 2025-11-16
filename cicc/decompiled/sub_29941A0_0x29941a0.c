// Function: sub_29941A0
// Address: 0x29941a0
//
__int64 __fastcall sub_29941A0(__int64 a1, char a2)
{
  unsigned __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r12

  v2 = **(_QWORD **)(a1 + 912) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (**(_QWORD **)(a1 + 912) & 4) == 0 )
  {
    sub_2990FA0(a1, **(_QWORD **)(a1 + 912) & 0xFFFFFFFFFFFFFFF8LL);
    if ( !a2 )
      return v2;
    v4 = sub_AA5190(v2);
    if ( v4 )
    {
      if ( v4 == v2 + 48 )
        return v2;
    }
  }
  v5 = sub_2993B60(a1, v2);
  sub_2993860(a1, *(__int64 **)(a1 + 912), v5, 1);
  *(_QWORD *)(a1 + 912) = sub_22DDF00(*(_QWORD **)(a1 + 40), v5);
  return v5;
}
