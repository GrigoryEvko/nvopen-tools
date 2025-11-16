// Function: sub_130F450
// Address: 0x130f450
//
__int64 __fastcall sub_130F450(__int64 a1, const char *a2)
{
  const char *v2; // rcx
  __int64 result; // rax
  int v4; // eax
  int v5; // r13d
  const char *v6; // r14
  int i; // r12d

  if ( *(_BYTE *)(a1 + 29) )
  {
    *(_BYTE *)(a1 + 29) = 0;
    v4 = *(_DWORD *)a1;
    goto LABEL_15;
  }
  if ( *(_BYTE *)(a1 + 28) )
  {
    sub_130F0B0(a1, ",");
    if ( *(_DWORD *)a1 == 1 )
      goto LABEL_4;
    goto LABEL_6;
  }
  if ( *(_DWORD *)a1 != 1 )
  {
LABEL_6:
    sub_130F0B0(a1, "\n");
    v4 = *(_DWORD *)a1;
    v5 = *(_DWORD *)(a1 + 24);
    if ( *(_DWORD *)a1 )
    {
      v5 *= 2;
      if ( v5 <= 0 )
        goto LABEL_15;
      v6 = " ";
    }
    else
    {
      v6 = "\t";
      if ( v5 <= 0 )
        goto LABEL_16;
    }
    for ( i = 0; i < v5; ++i )
      sub_130F0B0(a1, "%s", v6);
    v4 = *(_DWORD *)a1;
LABEL_15:
    if ( v4 == 1 )
      goto LABEL_4;
LABEL_16:
    v2 = " ";
    goto LABEL_5;
  }
LABEL_4:
  v2 = byte_3F871B3;
LABEL_5:
  result = sub_130F0B0(a1, "\"%s\":%s", a2, v2);
  *(_BYTE *)(a1 + 29) = 1;
  return result;
}
