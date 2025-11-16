// Function: sub_72C570
// Address: 0x72c570
//
__int64 sub_72C570()
{
  __int64 result; // rax
  __int64 v1; // rdi
  _QWORD *v2; // rdx
  __int64 v3; // rax

  result = qword_4F07B60;
  if ( !qword_4F07B60 )
  {
    qword_4F07B60 = (__int64)sub_7259C0(19);
    sub_8D6090(qword_4F07B60);
    v1 = qword_4F07B60;
    if ( !*(_QWORD *)(qword_4F07B60 - 16) )
    {
      v2 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
      v3 = v2[13];
      if ( qword_4F07B60 != v3 )
      {
        if ( v3 )
          *(_QWORD *)(v3 - 16) = qword_4F07B60;
        else
          v2[12] = qword_4F07B60;
        v2[13] = v1;
      }
    }
    sub_8CC670(v1);
    return qword_4F07B60;
  }
  return result;
}
