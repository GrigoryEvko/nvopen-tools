// Function: sub_5C8500
// Address: 0x5c8500
//
__int64 __fastcall sub_5C8500(__int64 a1, __int64 a2, char a3)
{
  __int64 i; // rax
  __int64 v5; // rbx
  const char *v6; // rax

  if ( a3 != 11 )
    return a2;
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(i + 168);
  sub_826A90(a2 + 64, a2);
  *(_BYTE *)(a2 + 199) |= 1u;
  if ( !v5 || (*(_BYTE *)(v5 + 16) & 1) == 0 )
  {
LABEL_10:
    v6 = *(const char **)(a2 + 8);
    if ( v6 )
      goto LABEL_11;
    return a2;
  }
  v6 = *(const char **)(a2 + 8);
  if ( !v6 )
    return a2;
  if ( !strcmp(*(const char **)(a2 + 8), "printf") )
  {
    *(_BYTE *)(v5 + 24) = 1;
    goto LABEL_10;
  }
LABEL_11:
  if ( strcmp(v6, "memcpy") && strcmp(v6, "memset") )
    return a2;
  *(_BYTE *)(a2 + 199) |= 2u;
  return a2;
}
