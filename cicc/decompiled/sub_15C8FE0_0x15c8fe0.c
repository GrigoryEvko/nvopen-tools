// Function: sub_15C8FE0
// Address: 0x15c8fe0
//
_QWORD *__fastcall sub_15C8FE0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *result; // rax
  bool v5; // zf
  __int64 v6; // rdx

  *(_BYTE *)(a1 + 12) = a4;
  *(_DWORD *)(a1 + 8) = 0;
  result = &unk_49ECEA0;
  v5 = *(_QWORD *)(a2 + 48) == 0;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = &unk_49ECEA0;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a2;
  if ( !v5 || *(__int16 *)(a2 + 18) < 0 )
  {
    result = (_QWORD *)sub_1625940(a2, "srcloc", 6);
    if ( result )
    {
      v6 = *((unsigned int *)result + 2);
      if ( (_DWORD)v6 )
      {
        result = (_QWORD *)result[-v6];
        if ( *(_BYTE *)result == 1 )
        {
          result = (_QWORD *)result[17];
          if ( *((_BYTE *)result + 16) == 13 )
          {
            if ( *((_DWORD *)result + 8) <= 0x40u )
              result = (_QWORD *)result[3];
            else
              result = *(_QWORD **)result[3];
            *(_DWORD *)(a1 + 16) = (_DWORD)result;
          }
        }
      }
    }
  }
  return result;
}
