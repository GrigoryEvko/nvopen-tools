// Function: sub_1628280
// Address: 0x1628280
//
__int64 __fastcall sub_1628280(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 result; // rax
  unsigned int v4; // ecx

  if ( !a3 )
    return sub_1627350(a1, a2, a3, 0, 1);
  result = *a2;
  if ( !*a2
    || (unsigned __int8)(*(_BYTE *)result - 4) > 0x1Eu
    || (__int64 *)*(unsigned int *)(result + 8) != a3
    || result != *(_QWORD *)(result - 8LL * (_QWORD)a3) )
  {
    return sub_1627350(a1, a2, a3, 0, 1);
  }
  if ( a3 != (__int64 *)1 )
  {
    v4 = 1;
    while ( a2[v4] == *(_QWORD *)(result + 8 * (v4 - (_QWORD)a3)) )
    {
      if ( (_DWORD)a3 == ++v4 )
        return result;
    }
    return sub_1627350(a1, a2, a3, 0, 1);
  }
  return result;
}
