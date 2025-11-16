// Function: sub_72A2A0
// Address: 0x72a2a0
//
__int64 __fastcall sub_72A2A0(__int64 a1)
{
  char v1; // al
  __int64 j; // rax
  unsigned int v4; // r8d
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 i; // rax
  unsigned __int8 v8; // r13

  if ( (*(_QWORD *)(a1 + 168) & 0xFF0000000008LL) == 0x10000000000LL )
    return (unsigned int)sub_6210B0(a1, 0) == 0;
  v1 = *(_BYTE *)(a1 + 173);
  if ( ((v1 - 3) & 0xFD) != 0 )
  {
    if ( v1 == 4 )
    {
      for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v8 = *(_BYTE *)(i + 160);
      v4 = sub_70B8A0(v8, *(const __m128i **)(a1 + 176));
      if ( v4 )
        return (unsigned int)sub_70B8A0(v8, (const __m128i *)(*(_QWORD *)(a1 + 176) + 16LL)) != 0;
    }
    else
    {
      v4 = 0;
      if ( v1 == 8 )
      {
        v5 = *(_QWORD *)(a1 + 176);
        if ( *(_BYTE *)(v5 + 173) == 6 && *(_BYTE *)(v5 + 176) == 6 )
        {
          v6 = *(_QWORD *)(a1 + 184);
          if ( *(_BYTE *)(v6 + 173) == 6 && *(_BYTE *)(v6 + 176) == 6 )
            return *(_QWORD *)(v5 + 184) == *(_QWORD *)(v6 + 184);
        }
      }
    }
    return v4;
  }
  else
  {
    for ( j = *(_QWORD *)(a1 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    return sub_70B8A0(*(_BYTE *)(j + 160), (const __m128i *)(a1 + 176));
  }
}
