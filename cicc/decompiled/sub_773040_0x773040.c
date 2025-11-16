// Function: sub_773040
// Address: 0x773040
//
__int64 __fastcall sub_773040(char *a1)
{
  char v1; // dl
  __int64 result; // rax
  __int64 v3; // rax
  char v4; // dl
  __int64 v5; // rax
  __int64 v6; // rdx
  char v7; // al

  v1 = *a1;
  if ( *a1 != 48 )
  {
    if ( v1 == 7 )
    {
      v5 = *(_QWORD *)(*((_QWORD *)a1 + 1) + 216LL);
      if ( !v5 )
        return 0;
      result = *(_QWORD *)(v5 + 16);
      if ( !result || *(_BYTE *)(result + 120) != 3 )
        return 0;
      return result;
    }
    if ( v1 == 11 )
    {
      result = *(_QWORD *)(*((_QWORD *)a1 + 1) + 248LL);
      if ( result && *(_BYTE *)(result + 120) == 2 )
        return result;
      return 0;
    }
    result = 0;
    if ( v1 != 6 )
      return result;
    v3 = *((_QWORD *)a1 + 1);
    v4 = *(_BYTE *)(v3 + 140);
    if ( (unsigned __int8)(v4 - 9) <= 2u )
      goto LABEL_9;
LABEL_22:
    if ( v4 != 12 )
      return 0;
    result = *(_QWORD *)(*(_QWORD *)(v3 + 168) + 16LL);
    goto LABEL_10;
  }
  v6 = *((_QWORD *)a1 + 1);
  v7 = *(_BYTE *)(v6 + 8);
  if ( v7 == 1 )
  {
    *a1 = 2;
    *((_QWORD *)a1 + 1) = *(_QWORD *)(v6 + 32);
    return 0;
  }
  if ( v7 == 2 )
  {
    *a1 = 59;
    *((_QWORD *)a1 + 1) = *(_QWORD *)(v6 + 32);
    return 0;
  }
  if ( v7 )
    sub_721090();
  *a1 = 6;
  v3 = *(_QWORD *)(v6 + 32);
  *((_QWORD *)a1 + 1) = v3;
  v4 = *(_BYTE *)(v3 + 140);
  if ( (unsigned __int8)(v4 - 9) > 2u )
    goto LABEL_22;
LABEL_9:
  result = *(_QWORD *)(*(_QWORD *)(v3 + 168) + 160LL);
LABEL_10:
  if ( !result || *(_BYTE *)(result + 120) != 1 )
    return 0;
  return result;
}
