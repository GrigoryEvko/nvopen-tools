// Function: sub_FC7530
// Address: 0xfc7530
//
__int64 __fastcall sub_FC7530(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a1[2];
  if ( a2 != v2 )
  {
    if ( v2 != 0 && v2 != -4096 && v2 != -8192 )
      sub_BD60C0(a1);
    a1[2] = a2;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      sub_BD73F0((__int64)a1);
  }
  return a2;
}
