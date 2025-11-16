// Function: sub_10E25C0
// Address: 0x10e25c0
//
__int64 __fastcall sub_10E25C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  result = 0;
  if ( *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v3 + 36) == *(_DWORD *)a1 )
      {
        v4 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( v4 )
        {
          **(_QWORD **)(a1 + 16) = v4;
          return 1;
        }
      }
    }
  }
  return result;
}
