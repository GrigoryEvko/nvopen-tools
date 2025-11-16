// Function: sub_72C4C0
// Address: 0x72c4c0
//
__int64 sub_72C4C0()
{
  __int64 result; // rax
  _QWORD *v1; // rax
  _QWORD *v2; // rdi
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  __int64 v5; // rax

  result = qword_4F07B58;
  if ( !qword_4F07B58 )
  {
    v1 = sub_7259C0(19);
    *((_BYTE *)v1 + 141) |= 0x20u;
    v2 = v1;
    qword_4F07B58 = (__int64)v1;
    if ( !*(v1 - 2) )
    {
      v3 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
      v4 = (_QWORD *)v3[13];
      if ( v2 != v4 )
      {
        if ( v4 )
        {
          *(v4 - 2) = v2;
          v5 = qword_4F07B58;
        }
        else
        {
          v3[12] = v2;
          v5 = (__int64)v2;
        }
        v3[13] = v2;
        v2 = (_QWORD *)v5;
      }
    }
    sub_8CC670(v2);
    return qword_4F07B58;
  }
  return result;
}
