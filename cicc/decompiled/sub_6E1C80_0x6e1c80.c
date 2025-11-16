// Function: sub_6E1C80
// Address: 0x6e1c80
//
__int64 *__fastcall sub_6E1C80(__int64 *a1)
{
  __int64 *v1; // r12
  __int64 v2; // rax
  __int64 v4; // rax

  if ( a1 )
  {
    v1 = (__int64 *)*a1;
    if ( !*a1 )
      return v1;
    v2 = *v1;
    if ( *v1 )
    {
      if ( *(_BYTE *)(v2 + 8) != 3 )
      {
        *a1 = v2;
LABEL_6:
        *v1 = 0;
        return v1;
      }
      v4 = sub_6BBB10((_QWORD *)*a1);
      *a1 = v4;
      if ( v4 )
        goto LABEL_6;
    }
    else
    {
      *a1 = 0;
    }
    a1[1] = 0;
    goto LABEL_6;
  }
  return 0;
}
