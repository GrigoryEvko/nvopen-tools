// Function: sub_72B690
// Address: 0x72b690
//
__int64 sub_72B690()
{
  __int64 result; // rax
  _QWORD *v1; // [rsp-10h] [rbp-10h]

  result = qword_4F07B40;
  if ( !qword_4F07B40 )
  {
    v1 = sub_7259C0(18);
    qword_4F07B40 = (__int64)v1;
    sub_8D6090(v1);
    return (__int64)v1;
  }
  return result;
}
