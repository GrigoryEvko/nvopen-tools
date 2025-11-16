// Function: sub_BD2C40
// Address: 0xbd2c40
//
_QWORD *__fastcall sub_BD2C40(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // r8

  v2 = 4LL * a2;
  v3 = (_QWORD *)sub_22077B0(v2 * 8 + a1);
  v4 = &v3[v2];
  for ( HIDWORD(v3[v2]) = HIDWORD(v3[v2]) & 0x38000000 | a2 & 0x7FFFFFF; v4 != v3; v3 += 4 )
  {
    if ( v3 )
    {
      *v3 = 0;
      v3[1] = 0;
      v3[2] = 0;
      v3[3] = v4;
    }
  }
  return v4;
}
