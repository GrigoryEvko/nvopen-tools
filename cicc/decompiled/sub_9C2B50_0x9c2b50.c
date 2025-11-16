// Function: sub_9C2B50
// Address: 0x9c2b50
//
unsigned __int64 __fastcall sub_9C2B50(__int64 **a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 result; // rax
  int v4; // edx
  unsigned __int64 v5; // rdx

  v1 = **a1;
  if ( !v1 || (v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL, v2 == v1 + 48) )
  {
    v4 = *(_DWORD *)a1[1];
    result = 0;
    if ( v4 )
    {
      result = *(_QWORD *)(a1[2][194] + 8LL * (unsigned int)(v4 - 1));
      if ( result )
      {
        v5 = *(_QWORD *)(result + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v5 || result + 48 == v5 )
          return 0;
        else
          return v5 - 24;
      }
    }
  }
  else
  {
    result = v2 - 24;
    if ( !v2 )
      return 0;
  }
  return result;
}
