// Function: sub_5C9390
// Address: 0x5c9390
//
__int64 __fastcall sub_5C9390(unsigned __int64 a1, __int64 *a2)
{
  unsigned __int8 v3; // bl
  __int64 v4; // rax
  char *v5; // r8
  const char *v6; // rdi
  char v7; // dl
  unsigned __int64 v9; // rax
  char *endptr[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = 1;
  v4 = *a2;
  v5 = (char *)(v4 + 1);
  endptr[0] = (char *)(*a2 + 1);
  if ( *(_BYTE *)(v4 + 1) != 45 )
  {
    v9 = strtoul((const char *)(v4 + 1), endptr, 10);
    v6 = endptr[0];
    v7 = *endptr[0];
    v3 = v9 <= a1;
    if ( *endptr[0] != 45 )
    {
LABEL_5:
      v3 &= v9 >= a1;
      goto LABEL_3;
    }
    v5 = endptr[0];
  }
  v6 = v5 + 1;
  endptr[0] = v5 + 1;
  v7 = v5[1];
  if ( (unsigned __int8)(v7 - 48) <= 9u )
  {
    v9 = strtoul(v6, endptr, 10);
    v6 = endptr[0];
    v7 = *endptr[0];
    goto LABEL_5;
  }
LABEL_3:
  *a2 = (__int64)&v6[v7 == 41];
  return v3;
}
