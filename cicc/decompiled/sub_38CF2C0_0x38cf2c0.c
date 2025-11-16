// Function: sub_38CF2C0
// Address: 0x38cf2c0
//
__int64 __fastcall sub_38CF2C0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v4; // r10

  v4 = a3;
  LODWORD(a3) = 0;
  if ( v4 )
    a3 = (_QWORD *)*v4;
  return sub_38CEAE0(a1, a2, (int)a3, (__int64)v4, a4, 0, 0);
}
