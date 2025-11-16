// Function: sub_B44680
// Address: 0xb44680
//
char __fastcall sub_B44680(__int64 a1)
{
  char *v1; // r13
  __int64 v2; // r12
  char *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r12
  char *v6; // r12
  char result; // al

  v1 = (char *)a1;
  v2 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v3 = *(char **)(a1 - 8);
    v1 = &v3[v2];
  }
  else
  {
    v3 = (char *)(a1 - v2);
  }
  v4 = v2 >> 5;
  v5 = v2 >> 7;
  if ( v5 )
  {
    v6 = &v3[128 * v5];
    while ( !(unsigned __int8)sub_BD36B0(*(_QWORD *)v3) )
    {
      if ( (unsigned __int8)sub_BD36B0(*((_QWORD *)v3 + 4)) )
        return v1 != v3 + 32;
      if ( (unsigned __int8)sub_BD36B0(*((_QWORD *)v3 + 8)) )
        return v1 != v3 + 64;
      if ( (unsigned __int8)sub_BD36B0(*((_QWORD *)v3 + 12)) )
        return v1 != v3 + 96;
      v3 += 128;
      if ( v6 == v3 )
      {
        v4 = (v1 - v3) >> 5;
        goto LABEL_14;
      }
    }
    return v1 != v3;
  }
LABEL_14:
  if ( v4 == 2 )
    goto LABEL_20;
  if ( v4 == 3 )
  {
    if ( (unsigned __int8)sub_BD36B0(*(_QWORD *)v3) )
      return v1 != v3;
    v3 += 32;
LABEL_20:
    if ( !(unsigned __int8)sub_BD36B0(*(_QWORD *)v3) )
    {
      v3 += 32;
      goto LABEL_22;
    }
    return v1 != v3;
  }
  if ( v4 != 1 )
    return 0;
LABEL_22:
  result = sub_BD36B0(*(_QWORD *)v3);
  if ( result )
    return v1 != v3;
  return result;
}
