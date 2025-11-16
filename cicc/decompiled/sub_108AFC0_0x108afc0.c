// Function: sub_108AFC0
// Address: 0x108afc0
//
__int64 __fastcall sub_108AFC0(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  int v5; // r15d

  v3 = a1 + 239;
  v4 = (_QWORD *)a1[241];
  if ( v4 == v3 )
  {
    v5 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      if ( v4[6] == a2 )
        break;
      v5 = v5 - 1431655765 * ((__int64)(v4[9] - v4[8]) >> 3) + 1;
      v4 = (_QWORD *)sub_220EEE0(v4);
    }
    while ( v3 != v4 );
  }
  return v5 * (*(_BYTE *)(a1[23] + 8LL) == 0 ? 6 : 10);
}
