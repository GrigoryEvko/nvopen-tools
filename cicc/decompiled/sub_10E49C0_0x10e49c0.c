// Function: sub_10E49C0
// Address: 0x10e49c0
//
__int64 __fastcall sub_10E49C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = 0;
  if ( *(_BYTE *)a2 == 85 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v4 + 36) == *(_DWORD *)a1 )
      {
        v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 16) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( v5 )
        {
          v2 = 1;
          **(_QWORD **)(a1 + 24) = v5;
        }
      }
    }
  }
  return v2;
}
