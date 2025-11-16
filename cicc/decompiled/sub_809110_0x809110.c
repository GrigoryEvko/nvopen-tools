// Function: sub_809110
// Address: 0x809110
//
__int64 sub_809110()
{
  __int64 *v0; // rbx
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 *v4; // rax

  v0 = (__int64 *)qword_4F18BE8;
  if ( qword_4F18BE8 )
  {
    result = *(_QWORD *)(qword_4F18BE8 + 8);
  }
  else
  {
    v4 = (__int64 *)sub_822B10(16);
    *v4 = 0;
    v0 = v4;
    result = sub_8237A0(2048);
    v0[1] = result;
  }
  v2 = *v0;
  qword_4F18BE0 = result;
  qword_4F18BE8 = v2;
  v3 = qword_4F18BF0;
  qword_4F18BF0 = (__int64)v0;
  *v0 = v3;
  return result;
}
