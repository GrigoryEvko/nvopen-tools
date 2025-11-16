// Function: sub_24F5180
// Address: 0x24f5180
//
__int64 __fastcall sub_24F5180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // rax

  v4 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)a2 == 85 )
  {
    v8 = *(_QWORD *)(a2 - 32);
    if ( v8 )
    {
      if ( !*(_BYTE *)v8
        && *(_QWORD *)(v8 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v8 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v8 + 36) - 60) <= 2 )
      {
        v4 = sub_AA56F0(*(_QWORD *)(a2 + 40));
      }
    }
  }
  if ( *(_BYTE *)a3 != 84 )
  {
    v5 = *(_QWORD *)(a3 + 40);
    if ( *(_BYTE *)a3 == 85 )
    {
      v7 = *(_QWORD *)(a3 - 32);
      if ( v7 )
      {
        if ( !*(_BYTE *)v7
          && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a3 + 80)
          && (*(_BYTE *)(v7 + 33) & 0x20) != 0
          && *(_DWORD *)(v7 + 36) == 62
          || !*(_BYTE *)v7
          && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a3 + 80)
          && (*(_BYTE *)(v7 + 33) & 0x20) != 0
          && *(_DWORD *)(v7 + 36) == 61 )
        {
          v5 = sub_AA54C0(*(_QWORD *)(a3 + 40));
        }
      }
    }
    return sub_24F96E0(a1, v4, v5);
  }
  if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) <= 1 )
  {
    v5 = *(_QWORD *)(a3 + 40);
    return sub_24F96E0(a1, v4, v5);
  }
  return 0;
}
