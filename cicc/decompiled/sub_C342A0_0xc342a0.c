// Function: sub_C342A0
// Address: 0xc342a0
//
__int64 __fastcall sub_C342A0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  unsigned int v3; // r15d
  __int64 v4; // r13
  unsigned int v5; // eax

  v2 = 0;
  *(_DWORD *)(a1 + 16) += a2;
  v3 = sub_C337D0(a1);
  v4 = sub_C33900(a1);
  v5 = sub_C45DF0(v4, v3);
  if ( a2 > v5 )
  {
    if ( a2 == v5 + 1 )
    {
      v2 = 2;
    }
    else if ( a2 > v3 << 6 || (v2 = 3, !(unsigned int)sub_C45D90(v4, a2 - 1)) )
    {
      v2 = 1;
    }
  }
  sub_C48220(v4, v3, a2);
  return v2;
}
