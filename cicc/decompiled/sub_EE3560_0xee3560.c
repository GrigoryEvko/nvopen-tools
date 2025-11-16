// Function: sub_EE3560
// Address: 0xee3560
//
__int64 __fastcall sub_EE3560(char **a1, __int64 *a2)
{
  char *v2; // r9
  char *v3; // rax
  __int64 v4; // rdx
  char v6; // r8
  char *v7; // rax
  __int64 v8; // rcx

  v2 = a1[1];
  v3 = *a1;
  if ( *a1 == v2 )
    return 1;
  v4 = *v3;
  if ( (unsigned __int8)(v4 - 48) > 9u && (unsigned __int8)(v4 - 65) > 0x19u )
    return 1;
  v6 = v4 - 48;
  v7 = v3 + 1;
  v8 = 0;
  if ( (unsigned __int8)(v4 - 48) <= 9u )
    goto LABEL_9;
  while ( (unsigned __int8)(v4 - 65) <= 0x19u )
  {
    *a1 = v7;
    v8 = v4 + 36 * v8 - 55;
    if ( v2 == v7 )
      break;
    while ( 1 )
    {
      v4 = *v7++;
      v6 = v4 - 48;
      if ( (unsigned __int8)(v4 - 48) > 9u )
        break;
LABEL_9:
      *a1 = v7;
      v8 = v6 + 36 * v8;
      if ( v2 == v7 )
        goto LABEL_10;
    }
  }
LABEL_10:
  *a2 = v8;
  return 0;
}
