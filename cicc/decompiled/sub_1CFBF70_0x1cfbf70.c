// Function: sub_1CFBF70
// Address: 0x1cfbf70
//
_QWORD *__fastcall sub_1CFBF70(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12

  v1 = sub_22077B0(864);
  v2 = (_QWORD *)v1;
  if ( v1 )
  {
    sub_1D0DA30(v1, *(_QWORD *)(a1 + 256));
    v2[102] = 0;
    *v2 = off_49F93A0;
    v2[83] = v2 + 85;
    v2[84] = 0x1000000000LL;
    v2[103] = 0;
    v2[104] = 0;
    v2[105] = 0;
    v2[106] = 0;
    v2[107] = 0;
  }
  return v2;
}
