// Function: sub_F04E10
// Address: 0xf04e10
//
__int64 __fastcall sub_F04E10(__int64 *a1, _DWORD *a2)
{
  char *v2; // rax
  _BYTE *v3; // rcx
  char v4; // dl
  char *i; // rax
  __int64 v6; // rcx
  int v7; // edx

  v2 = (char *)a1[1];
  if ( !v2 )
    return 1;
  v3 = (_BYTE *)*a1;
  v4 = *(_BYTE *)*a1;
  a1[1] = (__int64)(v2 - 1);
  *a1 = (__int64)(v3 + 1);
  if ( (unsigned __int8)(v4 - 48) > 9u )
    return 1;
  *a2 = (char)(v4 - 48);
  for ( i = (char *)a1[1]; i; i = (char *)a1[1] )
  {
    v7 = *(char *)*a1;
    if ( (unsigned __int8)(*(_BYTE *)*a1 - 48) > 9u )
      break;
    v6 = *a1 + 1;
    a1[1] = (__int64)(i - 1);
    *a1 = v6;
    *a2 = v7 + 10 * *a2 - 48;
  }
  return 0;
}
