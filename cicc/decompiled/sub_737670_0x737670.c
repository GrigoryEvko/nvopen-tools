// Function: sub_737670
// Address: 0x737670
//
__int64 __fastcall sub_737670(
        __int64 a1,
        unsigned __int8 a2,
        __int64 (__fastcall *a3)(__int64, _QWORD, _DWORD *),
        _DWORD *a4,
        int a5)
{
  unsigned __int8 v5; // r15
  __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // rdx

  v5 = a2;
  v9 = a1;
  if ( !a4 )
  {
    if ( (a5 & 8) != 0 )
      a3(a1, a2, 0);
    while ( 1 )
    {
LABEL_3:
      result = *(_QWORD *)(v9 + 40);
      if ( (!result || !*(_BYTE *)(result + 28)) && (*(_BYTE *)(v9 + 89) & 1) == 0 )
        return result;
      v11 = *(_QWORD *)(v9 + 48);
      if ( v11 )
      {
        v9 = *(_QWORD *)(v9 + 48);
        result = 4;
        v5 = 11;
      }
      else if ( (*(_BYTE *)(v9 + 89) & 4) != 0 )
      {
        v9 = *(_QWORD *)(result + 32);
        v5 = 6;
        result = 2;
      }
      else
      {
        if ( v5 == 6 )
        {
          if ( *(_BYTE *)(v9 + 140) == 14 )
            return result;
          if ( result )
          {
LABEL_17:
            if ( *(_BYTE *)(result + 28) == 3 )
              v11 = *(_QWORD *)(result + 32);
            v9 = v11;
            result = 1;
            v5 = 28;
            goto LABEL_8;
          }
        }
        else if ( result )
        {
          goto LABEL_17;
        }
        result = 1;
        v5 = 28;
        v9 = 0;
      }
LABEL_8:
      if ( (a5 & (unsigned int)result) != 0 )
        result = a3(v9, v5, a4);
      if ( a4 && *a4 )
        return result;
    }
  }
  *a4 = 0;
  if ( (a5 & 8) == 0 )
    goto LABEL_3;
  a3(a1, a2, a4);
  result = (unsigned int)*a4;
  if ( !(_DWORD)result )
    goto LABEL_3;
  return result;
}
