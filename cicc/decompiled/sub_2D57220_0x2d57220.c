// Function: sub_2D57220
// Address: 0x2d57220
//
char __fastcall sub_2D57220(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a1[2];
  if ( v2 != a2 )
  {
    if ( v2 != -4096 && v2 != 0 && v2 != -8192 )
      sub_BD60C0(a1);
    a1[2] = a2;
    LOBYTE(v2) = a2 != 0;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      LOBYTE(v2) = sub_BD73F0((__int64)a1);
  }
  return v2;
}
