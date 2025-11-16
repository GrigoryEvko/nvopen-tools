// Function: sub_E0E160
// Address: 0xe0e160
//
char *__fastcall sub_E0E160(__int64 a1)
{
  char *v1; // r10
  char v2; // r8
  char v3; // dl
  __int64 v4; // rcx
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  char *result; // rax
  char *v9; // rcx

  v1 = *(char **)a1;
  if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 <= 1u )
    return 0;
  v2 = *v1;
  v3 = 108;
  v4 = 62;
  v5 = 0;
  v6 = 62;
  while ( 1 )
  {
    v7 = v6 >> 1;
    if ( v3 < v2 )
    {
      v5 = v7 + 1;
      goto LABEL_8;
    }
    if ( v3 == v2 && byte_497A3E0[16 * v7 + 1] < v1[1] )
      break;
    v4 = v7;
    if ( v7 == v5 )
      goto LABEL_9;
LABEL_5:
    v6 = v4 + v5;
    v3 = byte_497A3E0[16 * ((v4 + v5) >> 1)];
  }
  v5 = v7 + 1;
LABEL_8:
  if ( v4 != v5 )
    goto LABEL_5;
LABEL_9:
  result = 0;
  v9 = &byte_497A3E0[16 * v4];
  if ( v2 == *v9 && v9[1] == v1[1] )
  {
    *(_QWORD *)a1 = v1 + 2;
    return v9;
  }
  return result;
}
