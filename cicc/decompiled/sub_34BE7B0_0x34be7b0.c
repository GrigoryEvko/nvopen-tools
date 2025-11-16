// Function: sub_34BE7B0
// Address: 0x34be7b0
//
__int64 __fastcall sub_34BE7B0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  int v4; // esi

  result = 0xFFFFFFFFLL;
  if ( *(_DWORD *)a1 >= *(_DWORD *)a2 )
  {
    result = 1;
    if ( *(_DWORD *)a1 <= *(_DWORD *)a2 )
    {
      v3 = a1[1];
      result = 0xFFFFFFFFLL;
      v4 = *(_DWORD *)(a2[1] + 24LL);
      if ( *(_DWORD *)(v3 + 24) >= v4 )
        return v4 < *(_DWORD *)(v3 + 24);
    }
  }
  return result;
}
