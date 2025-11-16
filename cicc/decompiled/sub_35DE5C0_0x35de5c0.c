// Function: sub_35DE5C0
// Address: 0x35de5c0
//
char __fastcall sub_35DE5C0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char result; // al

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 12 )
    return 0;
  if ( (unsigned __int8)sub_35DE150(a1, (char *)a2, a3, a4, a5) )
    return 0;
  result = sub_35DE500(a1, (char *)a2);
  if ( !result )
    return *(_BYTE *)a2 != 82 && *(_BYTE *)a2 > 0x1Cu;
  return result;
}
