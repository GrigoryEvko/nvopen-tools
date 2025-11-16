// Function: sub_2E78940
// Address: 0x2e78940
//
__int64 __fastcall sub_2E78940(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  _BYTE **v4; // rdx
  __int64 v5; // r13
  _BYTE *v6; // rdi
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 v9; // r13
  __int64 v10; // rdx

  result = sub_B2D610(a1, 55);
  if ( (_BYTE)result )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    {
      result = sub_B91C10(a1, 30);
      v3 = result;
      if ( result )
      {
        if ( *(_BYTE *)result == 5 )
        {
          result = *(unsigned __int8 *)(result - 16);
          if ( (result & 2) != 0 )
          {
            if ( *(_DWORD *)(v3 - 24) != 2 )
              return result;
            v4 = *(_BYTE ***)(v3 - 32);
            v5 = v3 - 16;
          }
          else
          {
            if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xF) != 2 )
              return result;
            v5 = v3 - 16;
            result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
            v4 = (_BYTE **)(v3 - 16 - result);
          }
          v6 = *v4;
          if ( *v4 )
          {
            if ( !*v6 )
            {
              result = sub_B91420((__int64)v6);
              if ( v7 == 17
                && !(*(_QWORD *)result ^ 0x732D656661736E75LL | *(_QWORD *)(result + 8) ^ 0x7A69732D6B636174LL)
                && *(_BYTE *)(result + 16) == 101 )
              {
                v8 = *(_BYTE *)(v3 - 16);
                v9 = (v8 & 2) != 0 ? *(_QWORD *)(v3 - 32) : v5 - 8LL * ((v8 >> 2) & 0xF);
                result = *(_QWORD *)(v9 + 8);
                if ( result )
                {
                  v10 = *(_QWORD *)(result + 136);
                  result = *(_QWORD *)(v10 + 24);
                  if ( *(_DWORD *)(v10 + 32) > 0x40u )
                    result = *(_QWORD *)result;
                  *(_QWORD *)(a2 + 688) = result;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
