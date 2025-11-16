// Function: sub_AC55A0
// Address: 0xac55a0
//
__int64 __fastcall sub_AC55A0(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  char *v4; // rax
  unsigned __int64 v5; // rdx

  v1 = sub_AC5570(a1, 8u);
  if ( !(_BYTE)v1 )
    return 0;
  v2 = v1;
  v4 = (char *)sub_AC52D0(a1);
  if ( v4[v5 - 1] )
    return 0;
  if ( v5 - 1 > v5 || v5 == 1 )
    return v2;
  LOBYTE(v2) = memchr(v4, 0, v5 - 1) == 0;
  return v2;
}
