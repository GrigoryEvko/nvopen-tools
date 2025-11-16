// Function: sub_2D00C30
// Address: 0x2d00c30
//
__int64 __fastcall sub_2D00C30(unsigned int *a1, _BYTE *a2)
{
  unsigned int *v3; // rax
  unsigned int *v4; // r8
  unsigned int *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx

  v3 = (unsigned int *)*((_QWORD *)a1 + 3);
  if ( v3 )
  {
    v4 = a1 + 4;
    v5 = a1 + 4;
    do
    {
      while ( 1 )
      {
        v6 = *((_QWORD *)v3 + 2);
        v7 = *((_QWORD *)v3 + 3);
        if ( *((_QWORD *)v3 + 4) >= (unsigned __int64)a2 )
          break;
        v3 = (unsigned int *)*((_QWORD *)v3 + 3);
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v3;
      v3 = (unsigned int *)*((_QWORD *)v3 + 2);
    }
    while ( v6 );
LABEL_6:
    if ( v4 != v5 && *((_QWORD *)v5 + 4) <= (unsigned __int64)a2 )
      return v5[10];
  }
  if ( *a2 != 5 )
    return a1[1];
  sub_2D00AD0(a1, (unsigned __int64)a2, *a1);
  return *a1;
}
