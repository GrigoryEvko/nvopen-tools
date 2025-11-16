// Function: sub_8C7520
// Address: 0x8c7520
//
_BOOL8 __fastcall sub_8C7520(__int64 **a1, __int64 **a2)
{
  __int64 *v2; // rcx
  _BOOL8 result; // rax
  __int64 *v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx

  if ( (*((_BYTE *)a1 + 89) & 8) != 0 )
  {
    v2 = a1[3];
    if ( (*((_BYTE *)a2 + 89) & 8) != 0 )
      goto LABEL_3;
LABEL_15:
    result = 1;
    if ( v2 == a2[1] )
      return result;
    goto LABEL_4;
  }
  v2 = a1[1];
  if ( (*((_BYTE *)a2 + 89) & 8) == 0 )
    goto LABEL_15;
LABEL_3:
  result = 1;
  if ( v2 == a2[3] )
    return result;
LABEL_4:
  v4 = *a1;
  v5 = *a2;
  if ( !*a1
    || !v5
    || (*((_BYTE *)v4 + 81) & 0x10) == 0 && sub_879510(*a1)
    || (*((_BYTE *)v5 + 81) & 0x10) == 0 && sub_879510(v5) )
  {
    return 0;
  }
  v6 = *v4;
  v7 = *v5;
  result = 1;
  if ( *v4 != *v5 )
  {
    result = 0;
    if ( *(_QWORD *)(v6 + 16) == *(_QWORD *)(v7 + 16) )
      return strncmp(*(const char **)(v6 + 8), *(const char **)(v7 + 8), *(_QWORD *)(v6 + 16)) == 0;
  }
  return result;
}
