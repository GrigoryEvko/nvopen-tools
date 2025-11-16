// Function: sub_8E7080
// Address: 0x8e7080
//
unsigned __int8 *__fastcall sub_8E7080(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  unsigned __int8 v4; // al
  __int64 v5; // rax
  __int64 v6; // rdx
  char *v7; // r8
  int v9; // edi
  __int64 v10; // [rsp+0h] [rbp-70h] BYREF
  __int64 v11; // [rsp+8h] [rbp-68h] BYREF
  char s[96]; // [rsp+10h] [rbp-60h] BYREF

  v2 = (unsigned __int8 *)(a1 + 1);
  v3 = *(_BYTE *)(a1 + 1);
  v10 = 1;
  v11 = -1;
  if ( v3 == 76 )
  {
    v2 = sub_8E5810((unsigned __int8 *)(a1 + 2), &v11, a2);
    if ( v11 < 0 )
      goto LABEL_20;
    ++v11;
    v3 = *v2;
  }
  if ( v3 != 112 )
  {
LABEL_20:
    if ( *(_DWORD *)(a2 + 24) )
      return v2;
    ++*(_QWORD *)(a2 + 32);
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
    return v2;
  }
  v4 = v2[1];
  if ( v4 == 84 )
  {
    v2 += 2;
    if ( !*(_QWORD *)(a2 + 32) )
      sub_8E5790((unsigned __int8 *)"this", a2);
    return v2;
  }
  ++v2;
  if ( v4 == 95 )
    goto LABEL_34;
  if ( (unsigned int)v4 - 48 <= 9 )
    goto LABEL_6;
  v9 = 0;
  while ( 1 )
  {
    if ( v4 == 75 )
    {
      v9 |= 1u;
      goto LABEL_27;
    }
    if ( v4 != 86 )
      break;
    v9 |= 2u;
LABEL_27:
    v4 = *++v2;
  }
  if ( v4 == 114 )
  {
    v9 |= 4u;
    goto LABEL_27;
  }
  sub_8E6E80(v9, 1, a2);
  if ( *v2 != 95 )
  {
LABEL_6:
    v2 = sub_8E5810(v2, &v10, a2);
    if ( v10 >= 0 )
    {
      v10 += 2;
      v5 = *(_QWORD *)(a2 + 32);
      v6 = v5;
      if ( *v2 == 95 )
        goto LABEL_35;
      if ( !*(_DWORD *)(a2 + 24) )
      {
        ++v5;
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
        *(_QWORD *)(a2 + 32) = v5;
      }
      goto LABEL_10;
    }
    goto LABEL_20;
  }
LABEL_34:
  v6 = *(_QWORD *)(a2 + 32);
LABEL_35:
  ++v2;
  v5 = v6;
LABEL_10:
  if ( !v5 )
    sub_8E5790("param#", a2);
  if ( v11 == -1 )
  {
    sprintf(s, "%ld", v10);
  }
  else
  {
    v7 = "s";
    if ( v11 <= 1 )
      v7 = (char *)byte_3F871B3;
    sprintf(s, "%ld[up %ld level%s]", v10, v11, v7);
  }
  if ( !*(_QWORD *)(a2 + 32) )
    sub_8E5790((unsigned __int8 *)s, a2);
  return v2;
}
