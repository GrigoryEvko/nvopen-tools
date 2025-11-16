// Function: sub_10A4F30
// Address: 0x10a4f30
//
_BOOL8 __fastcall sub_10A4F30(_QWORD **a1, _BYTE *a2)
{
  char v2; // al
  _BOOL8 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx

  v2 = *a2;
  if ( *a2 == 42 )
  {
    v4 = *((_QWORD *)a2 - 8);
    if ( !v4 )
      return 0;
    **a1 = v4;
    v5 = *((_QWORD *)a2 - 4);
    if ( v5 )
    {
      *a1[1] = v5;
      return 1;
    }
    v2 = *a2;
  }
  if ( v2 != 58 )
    return 0;
  result = (a2[1] & 2) != 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v6 = *((_QWORD *)a2 - 8);
  if ( !v6 )
    return 0;
  *a1[2] = v6;
  v7 = *((_QWORD *)a2 - 4);
  if ( !v7 )
    return 0;
  *a1[3] = v7;
  return result;
}
