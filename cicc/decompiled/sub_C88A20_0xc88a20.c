// Function: sub_C88A20
// Address: 0xc88a20
//
unsigned __int64 __fastcall sub_C88A20(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax

  v1 = a1[312];
  if ( v1 > 0x137 )
  {
    sub_C88920(a1);
    v1 = a1[312];
  }
  a1[312] = v1 + 1;
  v2 = (a1[v1] >> 29) & 0x5555555555555555LL ^ a1[v1];
  v3 = (((v2 << 17) & 0x71D67FFFEDA60000LL ^ v2) << 37) & 0xFFF7EEE000000000LL ^ (v2 << 17) & 0x71D67FFFEDA60000LL ^ v2;
  return (v3 >> 43) ^ v3;
}
