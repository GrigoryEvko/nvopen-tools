// Function: sub_ED7840
// Address: 0xed7840
//
__int64 __fastcall sub_ED7840(unsigned int *a1)
{
  unsigned int v1; // edx
  unsigned int v2; // esi
  unsigned __int64 v3; // rcx
  bool v4; // cf
  __int64 v5; // rax
  __int64 result; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = *((_QWORD *)a1 + 1);
  while ( 1 )
  {
    v4 = *(a1 - 4) < v1;
    if ( *(a1 - 4) <= v1 )
    {
      if ( *(a1 - 4) != v1 || (v4 = *(a1 - 3) < v2, *(a1 - 3) <= v2) )
      {
        if ( v4 )
        {
          *a1 = v1;
          a1[1] = v2;
          *((_QWORD *)a1 + 1) = v3;
          return result;
        }
        if ( *((_QWORD *)a1 - 1) <= v3 )
          break;
      }
    }
    v5 = *((_QWORD *)a1 - 2);
    a1 -= 4;
    *((_QWORD *)a1 + 2) = v5;
    result = *((_QWORD *)a1 + 1);
    *((_QWORD *)a1 + 3) = result;
  }
  *a1 = v1;
  a1[1] = v2;
  *((_QWORD *)a1 + 1) = v3;
  return result;
}
