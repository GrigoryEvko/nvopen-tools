// Function: sub_728BE0
// Address: 0x728be0
//
__int64 __fastcall sub_728BE0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 173);
  if ( !(_BYTE)result )
  {
    a2[20] = 1;
    a2[18] = 1;
    return result;
  }
  if ( (_BYTE)result == 12 )
  {
    result = *(unsigned __int8 *)(a1 + 176);
    if ( (_BYTE)result == 7 )
      goto LABEL_9;
    if ( (unsigned __int8)result <= 7u )
    {
      result = (unsigned int)(result - 5);
      if ( (unsigned __int8)result > 1u )
        return result;
      result = dword_4D047EC;
      if ( dword_4D047EC )
      {
        result = sub_8D4070(*(_QWORD *)(a1 + 184));
        if ( (_DWORD)result )
        {
          if ( !*(_QWORD *)(a1 + 192) && (*(_BYTE *)(a1 + 177) & 0x10) == 0 )
          {
            a2[20] = 1;
            a2[18] = 1;
            return result;
          }
        }
      }
      goto LABEL_9;
    }
    if ( (_BYTE)result == 10 )
LABEL_9:
      a2[19] = 1;
  }
  return result;
}
