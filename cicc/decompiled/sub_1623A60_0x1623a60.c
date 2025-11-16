// Function: sub_1623A60
// Address: 0x1623a60
//
__int64 __fastcall sub_1623A60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 result; // rax

  v4 = sub_161E620((unsigned __int8 *)a2);
  if ( v4 )
  {
    sub_16238A0(v4, a1, a3);
    return 1;
  }
  else
  {
    result = 0;
    if ( *(_BYTE *)a2 == 3 )
    {
      *(_QWORD *)(a2 + 8) = a1;
      return 1;
    }
  }
  return result;
}
