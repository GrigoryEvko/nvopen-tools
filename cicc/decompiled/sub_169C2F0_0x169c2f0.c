// Function: sub_169C2F0
// Address: 0x169c2f0
//
__int64 __fastcall sub_169C2F0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r12d
  int v4; // ebx
  int v5; // r12d
  __int16 *v6[2]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v7; // [rsp+10h] [rbp-30h]

  v1 = *(_BYTE *)(a1 + 18) & 7;
  if ( v1 == 1 )
    return 0x80000000;
  if ( v1 == 3 )
    return (unsigned int)-2147483647;
  v2 = 0x7FFFFFFF;
  if ( !v1 )
    return v2;
  if ( !sub_16984B0(a1) )
    return (unsigned int)*(__int16 *)(a1 + 16);
  sub_16986C0(v6, (__int64 *)a1);
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 4LL) - 1;
  v7 += v4;
  sub_1698EC0(v6, 0, 0);
  v5 = v7;
  sub_1698460((__int64)v6);
  return (unsigned int)(v5 - v4);
}
