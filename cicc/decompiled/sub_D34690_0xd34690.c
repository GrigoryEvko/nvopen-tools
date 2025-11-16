// Function: sub_D34690
// Address: 0xd34690
//
char __fastcall sub_D34690(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v3; // rbx
  unsigned int *v5; // rcx
  unsigned int *v6; // r13
  unsigned int *v7; // r15
  char result; // al
  unsigned int *v9; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int **)(a2 + 16);
  v9 = &v3[*(unsigned int *)(a2 + 24)];
  if ( v9 == v3 )
    return 0;
  while ( 1 )
  {
    v5 = *(unsigned int **)(a3 + 16);
    v6 = &v5[*(unsigned int *)(a3 + 24)];
    if ( v6 != v5 )
      break;
LABEL_7:
    if ( v9 == ++v3 )
      return 0;
  }
  v7 = *(unsigned int **)(a3 + 16);
  while ( 1 )
  {
    result = sub_D34650(a1, *v3, *v7);
    if ( result )
      return result;
    if ( v6 == ++v7 )
      goto LABEL_7;
  }
}
