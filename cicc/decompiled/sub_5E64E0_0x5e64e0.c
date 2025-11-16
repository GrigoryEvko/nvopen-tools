// Function: sub_5E64E0
// Address: 0x5e64e0
//
__int64 sub_5E64E0()
{
  __int64 result; // rax
  __int64 v1; // [rsp+8h] [rbp-8h] BYREF

  v1 = *(_QWORD *)&dword_4F063F8;
  if ( unk_4F077C4 == 2 && unk_4F07778 > 202301 )
    return sub_5CC190(24);
  if ( !dword_4F077B8 )
  {
    if ( !dword_4F077B4 )
      goto LABEL_4;
LABEL_10:
    if ( unk_4F077A0 > 0x1FBCFu )
      return sub_5CC190(24);
    goto LABEL_4;
  }
  if ( dword_4F077B4 )
    goto LABEL_10;
  if ( qword_4F077A8 > 0x1869Fu )
    return sub_5CC190(24);
LABEL_4:
  result = sub_5CC190(24);
  if ( result )
  {
    sub_684AA0(unk_4D04964 == 0 ? 5 : 7, 3226, &v1);
    return 0;
  }
  return result;
}
