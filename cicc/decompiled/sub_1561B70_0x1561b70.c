// Function: sub_1561B70
// Address: 0x1561b70
//
__int64 __fastcall sub_1561B70(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 result; // rax

  if ( *a2 != *a1 )
    return 0;
  v2 = a1[4];
  if ( a1 + 2 != (_QWORD *)v2 )
  {
    while ( a2 + 2 != (_QWORD *)sub_1561A70((__int64)(a2 + 1), v2 + 32) )
    {
      v2 = sub_220EF30(v2);
      if ( a1 + 2 == (_QWORD *)v2 )
        goto LABEL_3;
    }
    return 0;
  }
LABEL_3:
  if ( a1[7] != a2[7] )
    return 0;
  if ( a1[8] != a2[8] )
    return 0;
  result = 1;
  if ( a1[9] != a2[9] )
    return 0;
  return result;
}
