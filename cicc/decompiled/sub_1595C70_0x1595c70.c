// Function: sub_1595C70
// Address: 0x1595c70
//
__int64 __fastcall sub_1595C70(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  unsigned __int64 v4; // rdx
  char *v5; // r13
  signed __int64 v6; // r8
  _BYTE *v7; // rax

  v1 = sub_1595C40(a1, 8u);
  if ( !(_BYTE)v1 )
    return 0;
  v2 = v1;
  v5 = (char *)sub_1595920(a1);
  v6 = v4 - 1;
  if ( v5[v4 - 1] )
    return 0;
  if ( v4 < v6 || v4 == 1 )
    return v2;
  if ( v6 < 0 )
    v6 = 0x7FFFFFFFFFFFFFFFLL;
  v7 = memchr(v5, 0, v6);
  if ( !v7 )
    return v2;
  LOBYTE(v2) = v7 - v5 == -1;
  return v2;
}
