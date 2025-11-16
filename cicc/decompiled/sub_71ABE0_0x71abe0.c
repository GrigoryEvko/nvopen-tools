// Function: sub_71ABE0
// Address: 0x71abe0
//
__int64 __fastcall sub_71ABE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  __int64 v5; // [rsp+Ch] [rbp-34h] BYREF
  int v6; // [rsp+14h] [rbp-2Ch]
  __int64 v7; // [rsp+18h] [rbp-28h]

  v2 = *(__int64 **)(a1 + 72);
  v3 = *v2;
  if ( (unsigned int)sub_8DD3B0(*v2) )
    return 0;
  if ( *(_BYTE *)(a1 + 56) == 95 )
  {
    v3 = sub_8D46C0(*v2);
    if ( (unsigned int)sub_8DD3B0(v3) )
      return 0;
  }
  if ( !v3 )
    return 0;
  while ( *(_BYTE *)(v3 + 140) == 12 )
    v3 = *(_QWORD *)(v3 + 160);
  if ( (*(_BYTE *)(v3 + 141) & 0x20) != 0 || !(unsigned int)sub_8D4160(v3) )
    return 0;
  v6 = 0;
  v7 = 0;
  v5 = (unsigned int)sub_7A30C0(a1, 0, 1, a2);
  sub_67E3D0((__int64 *)((char *)&v5 + 4));
  return (unsigned int)v5;
}
