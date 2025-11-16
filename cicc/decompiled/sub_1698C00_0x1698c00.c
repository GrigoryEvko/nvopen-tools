// Function: sub_1698C00
// Address: 0x1698c00
//
__int64 __fastcall sub_1698C00(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  unsigned int v3; // r15d
  __int64 v4; // r13
  unsigned int v5; // eax

  v2 = 0;
  *(_WORD *)(a1 + 16) += a2;
  v3 = sub_1698310(a1);
  v4 = sub_1698470(a1);
  v5 = sub_16A7110(v4, v3);
  if ( a2 > v5 )
  {
    if ( a2 == v5 + 1 )
    {
      v2 = 2;
    }
    else if ( a2 > v3 << 6 || (v2 = 3, !(unsigned int)sub_16A70B0(v4, a2 - 1)) )
    {
      v2 = 1;
    }
  }
  sub_16A8050(v4, v3, a2);
  return v2;
}
