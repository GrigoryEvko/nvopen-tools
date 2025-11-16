// Function: sub_85FB90
// Address: 0x85fb90
//
int *__fastcall sub_85FB90(_QWORD *a1)
{
  int *result; // rax
  int v2; // edx

  result = &dword_4F04C64;
  v2 = dword_4F04C64;
  if ( dword_4F04C64 < 0 )
  {
    if ( dword_4F04C64 != -1 )
      return (int *)sub_85BC50(a1, *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 376));
  }
  else
  {
    result = (int *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    while ( *((_BYTE *)result + 4) != 9 || **((_QWORD ***)result + 51) != a1 )
    {
      --v2;
      result -= 194;
      if ( v2 == -1 )
        return result;
    }
    return (int *)sub_85BC50(a1, *((_QWORD *)result + 47));
  }
  return result;
}
