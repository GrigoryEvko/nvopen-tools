// Function: sub_10E4070
// Address: 0x10e4070
//
__int64 __fastcall sub_10E4070(_QWORD **a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = 0;
  if ( *a2 == 69 )
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
