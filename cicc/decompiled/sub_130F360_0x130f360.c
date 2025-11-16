// Function: sub_130F360
// Address: 0x130f360
//
__int64 __fastcall sub_130F360(__int64 a1)
{
  __int64 result; // rax
  int v2; // r13d
  const char *v3; // r14
  int i; // r12d

  if ( *(_BYTE *)(a1 + 29) )
  {
    *(_BYTE *)(a1 + 29) = 0;
    goto LABEL_4;
  }
  if ( *(_BYTE *)(a1 + 28) )
  {
    sub_130F0B0(a1, ",");
    if ( *(_DWORD *)a1 == 1 )
      goto LABEL_4;
  }
  else if ( *(_DWORD *)a1 == 1 )
  {
    goto LABEL_4;
  }
  sub_130F0B0(a1, "\n");
  v2 = *(_DWORD *)(a1 + 24);
  v3 = "\t";
  if ( *(_DWORD *)a1 )
  {
    v2 *= 2;
    v3 = " ";
  }
  if ( v2 > 0 )
  {
    for ( i = 0; i != v2; ++i )
      sub_130F0B0(a1, "%s", v3);
  }
LABEL_4:
  result = sub_130F0B0(a1, "{");
  ++*(_DWORD *)(a1 + 24);
  *(_BYTE *)(a1 + 28) = 0;
  return result;
}
