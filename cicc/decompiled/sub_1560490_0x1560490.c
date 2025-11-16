// Function: sub_1560490
// Address: 0x1560490
//
__int64 __fastcall sub_1560490(_QWORD *a1, char a2, int *a3)
{
  int v4; // eax
  int v5; // r13d
  int v6; // r15d
  __int64 result; // rax

  if ( !*a1 )
    return 0;
  v4 = sub_15601D0((__int64)a1);
  v5 = v4 - 1;
  if ( !v4 )
    return 0;
  v6 = -1;
  while ( 1 )
  {
    result = sub_1560260(a1, v6, a2);
    if ( (_BYTE)result )
      break;
    if ( ++v6 == v5 )
      return 0;
  }
  if ( a3 )
    *a3 = v6;
  return result;
}
