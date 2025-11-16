// Function: sub_CA8400
// Address: 0xca8400
//
char *__fastcall sub_CA8400(__int64 a1)
{
  char *result; // rax
  char v2; // al
  int v3; // edx

  for ( result = *(char **)(a1 + 40); ; result = *(char **)(a1 + 40) )
  {
LABEL_2:
    if ( *(char **)(a1 + 48) == result )
      goto LABEL_5;
    v2 = *result;
    if ( v2 != 32 )
      break;
LABEL_9:
    sub_CA7F70(a1, 1u);
  }
LABEL_4:
  if ( v2 == 9 )
    goto LABEL_9;
LABEL_5:
  while ( 1 )
  {
    sub_CA83A0(a1);
    result = sub_CA7C80(a1, *(_BYTE **)(a1 + 40));
    if ( *(char **)(a1 + 40) == result )
      return result;
    v3 = *(_DWORD *)(a1 + 68);
    ++*(_DWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 40) = result;
    *(_DWORD *)(a1 + 60) = 0;
    if ( v3 )
      goto LABEL_2;
    *(_BYTE *)(a1 + 73) = 1;
    if ( *(char **)(a1 + 48) != result )
    {
      v2 = *result;
      if ( v2 != 32 )
        goto LABEL_4;
      goto LABEL_9;
    }
  }
}
