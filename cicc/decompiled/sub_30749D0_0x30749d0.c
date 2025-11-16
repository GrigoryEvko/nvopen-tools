// Function: sub_30749D0
// Address: 0x30749d0
//
__int64 __fastcall sub_30749D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  unsigned int v4; // eax

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 344LL) > 0x45u && *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
      {
        v4 = *(_DWORD *)(v3 + 36);
        if ( v4 > 0x24F5 )
        {
          if ( v4 == 10649 )
            return 0;
        }
        else if ( v4 > 0x24E5 )
        {
          return 0;
        }
      }
    }
  }
  result = sub_A73ED0((_QWORD *)(a2 + 72), 6);
  if ( !(_BYTE)result )
    return sub_B49560(a2, 6);
  return result;
}
