// Function: sub_C57470
// Address: 0xc57470
//
__int64 *sub_C57470()
{
  if ( byte_4F83C48 )
    return &qword_4F83C60;
  if ( (unsigned int)sub_2207590(&byte_4F83C48) )
  {
    qword_4F83C68 = 15;
    qword_4F83C60 = (__int64)"General options";
    qword_4F83C70 = (__int64)byte_3F871B3;
    qword_4F83C78 = 0;
    sub_C524B0((__int64)&qword_4F83C60);
    sub_2207640(&byte_4F83C48);
  }
  return &qword_4F83C60;
}
