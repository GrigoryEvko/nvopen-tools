// Function: sub_2D56CB0
// Address: 0x2d56cb0
//
__int64 __fastcall sub_2D56CB0(__int64 a1, char *a2, char *a3, char *a4)
{
  __int64 v5; // rdi
  char v6; // al

  if ( a2 == a4 || a2 == 0 )
    return 1;
  v5 = (__int64)a2;
  if ( a3 == a2 )
    return 1;
  v6 = *a2;
  if ( (unsigned __int8)*a2 > 0x1Cu )
  {
    if ( v6 == 60 )
    {
      v5 = (__int64)a2;
      if ( sub_B4D040((__int64)a2) )
        return 1;
    }
  }
  else if ( v6 != 22 )
  {
    return 1;
  }
  return sub_BD37C0(v5, *(_QWORD *)(*(_QWORD *)(a1 + 88) + 40LL));
}
