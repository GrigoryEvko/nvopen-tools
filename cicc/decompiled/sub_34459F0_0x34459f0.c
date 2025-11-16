// Function: sub_34459F0
// Address: 0x34459f0
//
__int64 __fastcall sub_34459F0(__int64 *a1, unsigned int a2, char a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 result; // rax

  v5 = *((_DWORD *)a1 + 2);
  v6 = *a1;
  result = 1LL << a2;
  if ( a3 )
  {
    if ( v5 > 0x40 )
    {
      *(_QWORD *)(v6 + 8LL * (a2 >> 6)) |= result;
    }
    else
    {
      result |= v6;
      *a1 = result;
    }
  }
  else
  {
    result = ~result;
    if ( v5 > 0x40 )
    {
      *(_QWORD *)(v6 + 8LL * (a2 >> 6)) &= result;
    }
    else
    {
      result &= v6;
      *a1 = result;
    }
  }
  return result;
}
