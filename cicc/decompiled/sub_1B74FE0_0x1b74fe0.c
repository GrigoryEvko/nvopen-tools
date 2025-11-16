// Function: sub_1B74FE0
// Address: 0x1b74fe0
//
__int64 __fastcall sub_1B74FE0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a1[2];
  if ( a2 != v2 )
  {
    if ( v2 != 0 && v2 != -8 && v2 != -16 )
      sub_1649B30(a1);
    a1[2] = a2;
    if ( a2 != 0 && a2 != -8 && a2 != -16 )
      sub_164C220((__int64)a1);
  }
  return a2;
}
