// Function: sub_16C5C30
// Address: 0x16c5c30
//
unsigned __int64 __fastcall sub_16C5C30(char a1, __int64 a2)
{
  const char **v2; // r13
  char *v3; // rax
  char *v4; // r12
  size_t v5; // rax
  __int64 v6; // rdi
  size_t v7; // r13
  unsigned __int64 result; // rax
  __int64 v9; // rdx
  _QWORD v10[4]; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-20h] BYREF

  *(_DWORD *)(a2 + 8) = 0;
  if ( !a1 )
  {
    result = *(unsigned int *)(a2 + 12);
    v9 = 0;
    if ( result > 3 )
    {
LABEL_11:
      *(_DWORD *)(*(_QWORD *)a2 + v9) = 1886221359;
      *(_DWORD *)(a2 + 8) += 4;
      return result;
    }
LABEL_13:
    result = sub_16CD150(a2, a2 + 16, v9 + 4, 1);
    v9 = *(unsigned int *)(a2 + 8);
    goto LABEL_11;
  }
  v10[1] = "TMP";
  v2 = (const char **)v10;
  v10[2] = "TEMP";
  v10[0] = "TMPDIR";
  v10[3] = "TEMPDIR";
  v3 = getenv("TMPDIR");
  v4 = v3;
  if ( !v3 )
  {
    while ( &v11 != (char *)++v2 )
    {
      v3 = getenv(*v2);
      v4 = v3;
      if ( v3 )
        goto LABEL_5;
    }
    v9 = *(unsigned int *)(a2 + 8);
    result = *(unsigned int *)(a2 + 12) - v9;
    if ( result > 3 )
      goto LABEL_11;
    goto LABEL_13;
  }
LABEL_5:
  v5 = strlen(v3);
  v6 = *(unsigned int *)(a2 + 8);
  v7 = v5;
  result = *(unsigned int *)(a2 + 12) - v6;
  if ( v7 > result )
  {
    result = sub_16CD150(a2, a2 + 16, v7 + v6, 1);
    v6 = *(unsigned int *)(a2 + 8);
  }
  if ( v7 )
  {
    result = (unsigned __int64)memcpy((void *)(*(_QWORD *)a2 + v6), v4, v7);
    LODWORD(v6) = *(_DWORD *)(a2 + 8);
  }
  *(_DWORD *)(a2 + 8) = v6 + v7;
  return result;
}
