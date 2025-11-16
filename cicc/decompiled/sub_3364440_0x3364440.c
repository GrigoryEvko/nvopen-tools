// Function: sub_3364440
// Address: 0x3364440
//
char __fastcall sub_3364440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  char result; // al
  __int64 v10; // rdx
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  result = sub_3364290(a1, a4, a2, v11);
  if ( result )
  {
    result = 0;
    if ( v11[0] >= 0 )
    {
      v10 = 8 * v11[0];
      *a6 = 8 * v11[0];
      return a5 + v10 <= a3;
    }
  }
  return result;
}
