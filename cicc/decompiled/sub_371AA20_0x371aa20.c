// Function: sub_371AA20
// Address: 0x371aa20
//
__int64 __fastcall sub_371AA20(_BYTE *a1, _BYTE *a2)
{
  unsigned __int8 v2; // al
  bool v3; // cf
  __int64 result; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx

  v2 = a2[32];
  v3 = a1[32] < v2;
  if ( a1[32] != v2 )
    return v3 ? 1 : -1;
  v5 = *((_QWORD *)a2 + 1);
  v3 = *((_QWORD *)a1 + 1) < v5;
  if ( *((_QWORD *)a1 + 1) != v5 )
    return v3 ? 1 : -1;
  v6 = *((_QWORD *)a2 + 3);
  result = 0;
  if ( *((_QWORD *)a1 + 3) != v6 )
    return *((_QWORD *)a1 + 3) < v6 ? -1 : 1;
  return result;
}
