// Function: sub_7F6F90
// Address: 0x7f6f90
//
void __fastcall sub_7F6F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v7; // [rsp+8h] [rbp-18h]

  if ( !a2 || !a3 )
  {
    if ( a2 != a3 )
      goto LABEL_4;
LABEL_10:
    sub_7F51D0(a1, 1, 0, a4);
    return;
  }
  if ( *(_QWORD *)(a2 + 184) == *(_QWORD *)(a3 + 184) )
    goto LABEL_10;
LABEL_4:
  if ( a1 )
  {
    if ( *a5 )
    {
      v7 = a4;
      sub_7408F0(*a5, a1, a3, a4, (__int64)a5, a6);
      a4 = v7;
    }
    *a5 = a1;
  }
  sub_7F51D0(0, 1, 0, a4);
}
