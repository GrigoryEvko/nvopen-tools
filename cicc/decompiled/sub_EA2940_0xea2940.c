// Function: sub_EA2940
// Address: 0xea2940
//
__int64 __fastcall sub_EA2940(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  char v4; // dl

  v2 = a2[1];
  if ( a1[1] >= v2 )
  {
    if ( a1[1] > v2 )
      return 1;
    v3 = *(int *)a1;
    v4 = byte_3F85308[*(int *)a2];
    if ( (char)byte_3F85308[v3] <= v4 )
    {
      if ( (char)byte_3F85308[v3] >= v4 )
        BUG();
      return 1;
    }
  }
  return 0xFFFFFFFFLL;
}
