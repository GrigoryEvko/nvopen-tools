// Function: sub_7DAF40
// Address: 0x7daf40
//
__int64 sub_7DAF40()
{
  __int64 result; // rax
  _QWORD *v1; // rsi

  result = qword_4F18928;
  if ( !qword_4F18928 )
  {
    qword_4F18928 = sub_7E16B0(10);
    sub_7E1CA0(qword_4F18928);
    sub_72BA30(unk_4F0694C);
    sub_7E1B70("handle");
    sub_72BA30(byte_4F06A51[0]);
    sub_7E1B70("elem_size");
    v1 = sub_72BA30(7u);
    sub_7E1B70("elem_count");
    sub_7E1C00(qword_4F18928, (unsigned __int64)v1);
    return qword_4F18928;
  }
  return result;
}
