// Function: sub_EE3480
// Address: 0xee3480
//
__int64 __fastcall sub_EE3480(__int64 a1)
{
  char *v1; // rax
  unsigned int v2; // r10d
  char *v4; // rax
  char *v5; // rax

  v1 = *(char **)a1;
  v2 = 1;
  if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
    return 1;
  if ( *v1 == 104 )
  {
    *(_QWORD *)a1 = v1 + 1;
    if ( !sub_EE32C0((char **)a1, 1) )
      return v2;
    v5 = *(char **)a1;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v5 != 95 )
      return v2;
    goto LABEL_12;
  }
  if ( *v1 != 118 )
    return v2;
  *(_QWORD *)a1 = v1 + 1;
  if ( sub_EE32C0((char **)a1, 1) )
  {
    v4 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v4 == 95 )
    {
      *(_QWORD *)a1 = v4 + 1;
      if ( sub_EE32C0((char **)a1, 1) )
      {
        v5 = *(char **)a1;
        if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v5 == 95 )
        {
LABEL_12:
          v2 = 0;
          *(_QWORD *)a1 = v5 + 1;
          return v2;
        }
      }
    }
  }
  return 1;
}
