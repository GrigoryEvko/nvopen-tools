// Function: sub_16CFA40
// Address: 0x16cfa40
//
unsigned __int64 __fastcall sub_16CFA40(__int64 *a1, unsigned __int64 a2, int a3)
{
  _QWORD *v3; // r13
  unsigned __int64 v4; // rax
  unsigned int v5; // r12d
  unsigned __int64 v6; // rbx
  _QWORD v8[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( !a3 )
    a3 = sub_16CE270(a1, a2);
  v3 = (_QWORD *)(*a1 + 24LL * (unsigned int)(a3 - 1));
  v4 = *(_QWORD *)(*v3 + 16LL) - *(_QWORD *)(*v3 + 8LL);
  if ( v4 <= 0xFF )
  {
    v5 = sub_16CF450(v3, a2);
  }
  else if ( v4 <= 0xFFFF )
  {
    v5 = sub_16CF5C0(v3, a2);
  }
  else if ( v4 > 0xFFFFFFFF )
  {
    v5 = sub_16CF8C0(v3, a2);
  }
  else
  {
    v5 = sub_16CF740(v3, a2);
  }
  v6 = a2 - *(_QWORD *)(*v3 + 8LL);
  v8[0] = *(_QWORD *)(*v3 + 8LL);
  v8[1] = v6;
  return ((unsigned __int64)((unsigned int)v6 - (unsigned int)sub_16D25A0(v8, "\n\r", 2, -1)) << 32) | v5;
}
