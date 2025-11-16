// Function: sub_10A3FA0
// Address: 0x10a3fa0
//
__int64 __fastcall sub_10A3FA0(_QWORD **a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = 0;
  if ( *a2 == 68 )
  {
    v3 = *((_QWORD *)a2 - 4);
    if ( v3 )
    {
      **a1 = v3;
      return 1;
    }
  }
  return result;
}
