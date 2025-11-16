// Function: sub_72CBE0
// Address: 0x72cbe0
//
__int64 sub_72CBE0()
{
  __int64 result; // rax
  __int64 v1; // rdi
  _QWORD *v2; // rdx
  __int64 v3; // rax

  result = qword_4F07B98;
  if ( !qword_4F07B98 )
  {
    qword_4F07B98 = (__int64)sub_7259C0(1);
    v1 = qword_4F07B98;
    if ( !*(_QWORD *)(qword_4F07B98 - 16) )
    {
      v2 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
      v3 = v2[13];
      if ( qword_4F07B98 != v3 )
      {
        if ( v3 )
          *(_QWORD *)(v3 - 16) = qword_4F07B98;
        else
          v2[12] = qword_4F07B98;
        v2[13] = v1;
      }
    }
    sub_8CC670(v1);
    return qword_4F07B98;
  }
  return result;
}
