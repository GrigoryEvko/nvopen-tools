// Function: sub_14A9D60
// Address: 0x14a9d60
//
__int64 __fastcall sub_14A9D60(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 result; // rax

  v2 = *a1;
  result = 1LL << a2;
  if ( *((_DWORD *)a1 + 2) > 0x40u )
  {
    *(_QWORD *)(v2 + 8LL * (a2 >> 6)) |= result;
  }
  else
  {
    result |= v2;
    *a1 = result;
  }
  return result;
}
