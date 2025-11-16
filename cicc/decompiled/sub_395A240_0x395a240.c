// Function: sub_395A240
// Address: 0x395a240
//
bool __fastcall sub_395A240(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  unsigned int v6; // r15d
  unsigned int v7; // r15d
  bool result; // al
  int v9; // [rsp-3Ch] [rbp-3Ch] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11 )
    return 0;
  v6 = *(_DWORD *)(*(_QWORD *)a2 + 8LL);
  v7 = (v6 >> 8) - sub_14C23D0(a2, a1, 0, 0, 0, 0);
  result = 1;
  if ( v7 >= a3 )
  {
    if ( v7 > a3 )
    {
      return 0;
    }
    else
    {
      result = a4 == 2;
      if ( !a4 )
      {
        v9 = 1;
        return !sub_395A170(a1, a2, &v9);
      }
    }
  }
  return result;
}
