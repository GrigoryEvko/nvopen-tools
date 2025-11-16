// Function: sub_1CCAA20
// Address: 0x1ccaa20
//
__int64 __fastcall sub_1CCAA20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rax

  result = 0;
  if ( a1 )
  {
    if ( *(_BYTE *)(a1 + 16) == 71 )
    {
      v4 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v4 + 16) == 78 )
      {
        v5 = *(_QWORD *)(v4 - 24);
        if ( !*(_BYTE *)(v5 + 16) && *(_DWORD *)(v5 + 36) == 3956 )
        {
          v6 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(v6 + 16) == 86 )
          {
            v7 = *(_QWORD *)(v6 - 24);
            if ( *(_BYTE *)(v7 + 16) == 78 )
            {
              v8 = *(_QWORD *)(v7 - 24);
              if ( !*(_BYTE *)(v8 + 16) && *(_DWORD *)(v8 + 36) == 3785 )
              {
                v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)a2 + 32LL) + 32LL) + 80LL) + 32LL);
                *(_QWORD *)a3 = *(_QWORD *)(v9 + 120);
                *(_DWORD *)(a3 + 8) = *(_DWORD *)(v9 + 128);
                return 1;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
