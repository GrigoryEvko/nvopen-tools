// Function: sub_11BE170
// Address: 0x11be170
//
__int64 __fastcall sub_11BE170(__int64 *a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  result = *a2;
  if ( (unsigned __int8)result > 0x1Cu )
  {
    if ( (_BYTE)result != 63 )
      return result;
LABEL_6:
    v3 = (1LL << sub_BB5560((__int64)a2, a1[1])) | *(_QWORD *)(*a1 + 8);
    result = -v3 & v3;
    *(_QWORD *)(*a1 + 8) = result;
    return result;
  }
  if ( (_BYTE)result == 5 && *((_WORD *)a2 + 1) == 34 )
    goto LABEL_6;
  return result;
}
