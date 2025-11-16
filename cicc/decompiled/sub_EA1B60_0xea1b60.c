// Function: sub_EA1B60
// Address: 0xea1b60
//
__int64 __fastcall sub_EA1B60(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  while ( 1 )
  {
    result = *a2;
    if ( (_BYTE)result == 2 )
      break;
    if ( (_BYTE)result == 3 )
    {
      a2 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
    }
    else
    {
      if ( (_BYTE)result )
        return result;
      sub_EA1B60(a1, *((_QWORD *)a2 + 2));
      a2 = (unsigned __int8 *)*((_QWORD *)a2 + 3);
    }
  }
  result = *(_DWORD *)a2 >> 8;
  if ( (_WORD)result == 125 || (_WORD)result == 128 )
  {
    sub_E5CB20(*(_QWORD *)(a1 + 296), *((_QWORD *)a2 + 2), a3, a4, a5, a6);
    result = *((_QWORD *)a2 + 2);
    *(_WORD *)(result + 12) |= 0x100u;
  }
  return result;
}
