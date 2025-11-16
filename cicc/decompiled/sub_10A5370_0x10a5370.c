// Function: sub_10A5370
// Address: 0x10a5370
//
__int64 __fastcall sub_10A5370(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 85 )
    {
      v5 = *(_QWORD *)(a2 - 32);
      if ( v5 )
      {
        if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v5 + 36) == *(_DWORD *)a1 )
        {
          v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
          if ( *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v6)) )
          {
            v7 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - v6));
            if ( v7 )
            {
              v3 = 1;
              **(_QWORD **)(a1 + 32) = v7;
            }
          }
        }
      }
    }
  }
  return v3;
}
