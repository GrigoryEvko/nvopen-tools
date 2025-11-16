// Function: sub_1698320
// Address: 0x1698320
//
__int64 __fastcall sub_1698320(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  *a1 = a2;
  result = sub_1698310((__int64)a1);
  if ( (unsigned int)result > 1 )
  {
    result = sub_2207820(8LL * (unsigned int)result);
    a1[1] = result;
  }
  return result;
}
