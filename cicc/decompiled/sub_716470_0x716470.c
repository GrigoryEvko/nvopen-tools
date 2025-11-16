// Function: sub_716470
// Address: 0x716470
//
__int64 __fastcall sub_716470(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  result = qword_4F078B8;
  a2[1] = a1;
  *a2 = result;
  qword_4F078B8 = (__int64)a2;
  return result;
}
