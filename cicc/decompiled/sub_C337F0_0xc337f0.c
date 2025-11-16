// Function: sub_C337F0
// Address: 0xc337f0
//
__int64 __fastcall sub_C337F0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  *a1 = a2;
  result = sub_C337D0((__int64)a1);
  if ( (unsigned int)result > 1 )
  {
    result = sub_2207820(8LL * (unsigned int)result);
    a1[1] = result;
  }
  return result;
}
