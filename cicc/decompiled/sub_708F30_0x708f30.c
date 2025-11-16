// Function: sub_708F30
// Address: 0x708f30
//
__int64 sub_708F30()
{
  __int64 result; // rax
  __int64 v1; // r12

  if ( unk_4F074B0 )
  {
    dword_4F06C5C = 0;
    return (__int64)&dword_4F06C5C;
  }
  v1 = *(_QWORD *)(unk_4D03FF0 + 8LL);
  if ( dword_4F077C4 != 2 )
  {
    result = dword_4F06C5C;
    if ( !dword_4F06C5C )
      return result;
LABEL_6:
    sub_734F30();
    return sub_735400(v1);
  }
  sub_735280(v1);
  result = dword_4F06C5C;
  if ( dword_4F06C5C )
    goto LABEL_6;
  return result;
}
