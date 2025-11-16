// Function: sub_1698CF0
// Address: 0x1698cf0
//
__int64 __fastcall sub_1698CF0(__int64 a1, __int64 a2)
{
  int v2; // eax
  _BOOL4 v3; // r8d
  unsigned int v5; // r14d
  __int64 v6; // r13
  __int64 v7; // rax

  v2 = *(__int16 *)(a1 + 16) - *(__int16 *)(a2 + 16);
  if ( !v2 )
  {
    v5 = sub_1698310(a1);
    v6 = sub_16984A0(a2);
    v7 = sub_16984A0(a1);
    v2 = sub_16A98D0(v7, v6, v5);
  }
  v3 = v2 == 0;
  if ( v2 > 0 )
    return 2;
  return v3;
}
