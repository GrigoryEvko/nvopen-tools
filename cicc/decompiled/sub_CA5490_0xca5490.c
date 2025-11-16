// Function: sub_CA5490
// Address: 0xca5490
//
__int64 *sub_CA5490()
{
  if ( byte_4F84FC8 )
    return &qword_4F84FE0;
  if ( (unsigned int)sub_2207590(&byte_4F84FC8) )
  {
    qword_4F84FE8 = 13;
    qword_4F84FE0 = (__int64)"Color Options";
    qword_4F84FF0 = (__int64)byte_3F871B3;
    qword_4F84FF8 = 0;
    sub_C524B0((__int64)&qword_4F84FE0);
    sub_2207640(&byte_4F84FC8);
  }
  return &qword_4F84FE0;
}
