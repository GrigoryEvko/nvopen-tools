// Function: sub_13D24E0
// Address: 0x13d24e0
//
bool __fastcall sub_13D24E0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  unsigned int v3; // ecx
  bool result; // al
  int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 <= 0x17u )
  {
    result = 0;
    if ( v2 == 5 )
    {
      v5 = *(unsigned __int16 *)(a2 + 18);
      v6 = v5 - 17;
      result = (unsigned int)(v5 - 17) <= 1 || (unsigned __int16)(v5 - 24) <= 1u;
      if ( result )
      {
        result = (*(_BYTE *)(a2 + 17) & 2) != 0;
        if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
        {
          result = 0;
          if ( v6 <= 1 )
          {
            v7 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            if ( v7 )
            {
              **(_QWORD **)a1 = v7;
              return *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == *(_QWORD *)(a1 + 8);
            }
          }
        }
      }
    }
  }
  else
  {
    v3 = v2 - 41;
    result = v3 <= 1 || (unsigned __int8)(v2 - 48) <= 1u;
    if ( result )
    {
      result = (*(_BYTE *)(a2 + 17) & 2) != 0;
      if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
      {
        result = 0;
        if ( v3 <= 1 )
        {
          v8 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
             ? *(__int64 **)(a2 - 8)
             : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          v9 = *v8;
          result = 0;
          if ( v9 )
          {
            **(_QWORD **)a1 = v9;
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v10 = *(_QWORD *)(a2 - 8);
            else
              v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            return *(_QWORD *)(v10 + 24) == *(_QWORD *)(a1 + 8);
          }
        }
      }
    }
  }
  return result;
}
