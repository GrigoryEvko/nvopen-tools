// Function: sub_72B900
// Address: 0x72b900
//
__int64 sub_72B900()
{
  __int64 i; // r15
  __int64 v1; // rbx
  __int64 result; // rax

  for ( i = 0; i != 13; ++i )
  {
    qword_4F07EC0[i] = sub_8C7330((unsigned int)i);
    qword_4F07E40[i] = sub_8C7360((unsigned int)i);
    if ( unk_4D04548 | unk_4D04558 )
    {
      qword_4F07DC0[i] = sub_8C7390((unsigned int)i);
      qword_4F07D40[i] = sub_8C73A0((unsigned int)i);
    }
  }
  v1 = 0;
  qword_4F07B88 = sub_8C73B0();
  qword_4F07B80 = sub_8C73D0();
  qword_4F07B78 = sub_8C73F0();
  qword_4F07B70 = sub_8C7410();
  qword_4F07B60 = sub_8C7450();
  qword_4F07B58 = sub_8C7430();
  qword_4F07B68 = sub_8C74A0();
  do
  {
    qword_4F07CC0[v1] = sub_8C7470((unsigned int)v1);
    qword_4F07C40[v1] = sub_8C74C0((unsigned int)v1);
    result = sub_8C74F0((unsigned int)v1);
    qword_4F07BC0[v1++] = result;
  }
  while ( v1 != 14 );
  return result;
}
