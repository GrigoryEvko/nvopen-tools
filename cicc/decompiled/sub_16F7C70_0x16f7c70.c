// Function: sub_16F7C70
// Address: 0x16f7c70
//
char *__fastcall sub_16F7C70(__int64 a1)
{
  char *result; // rax
  char v2; // al
  int v3; // edx

  result = *(char **)(a1 + 40);
LABEL_2:
  while ( 1 )
  {
    v2 = *result;
    if ( v2 != 32 )
      break;
LABEL_7:
    sub_16F7930(a1, 1u);
    result = *(char **)(a1 + 40);
  }
  while ( 1 )
  {
    if ( v2 == 9 )
      goto LABEL_7;
    sub_16F7C20(a1);
    result = sub_16F7720(a1, *(_BYTE **)(a1 + 40));
    if ( *(char **)(a1 + 40) == result )
      return result;
    v3 = *(_DWORD *)(a1 + 68);
    ++*(_DWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 40) = result;
    *(_DWORD *)(a1 + 60) = 0;
    if ( v3 )
      goto LABEL_2;
    *(_BYTE *)(a1 + 73) = 1;
    v2 = *result;
    if ( v2 == 32 )
      goto LABEL_7;
  }
}
