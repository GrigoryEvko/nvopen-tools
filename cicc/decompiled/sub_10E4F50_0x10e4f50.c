// Function: sub_10E4F50
// Address: 0x10e4f50
//
__int64 __fastcall sub_10E4F50(_QWORD **a1, _BYTE *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdx

  if ( *a2 != 68 )
  {
    if ( *a2 == 69 )
    {
      v3 = *((_QWORD *)a2 - 4);
      if ( v3 )
      {
        *a1[1] = v3;
        return 1;
      }
    }
LABEL_3:
    *a1[2] = a2;
    return 1;
  }
  v4 = *((_QWORD *)a2 - 4);
  if ( !v4 )
    goto LABEL_3;
  **a1 = v4;
  return 1;
}
