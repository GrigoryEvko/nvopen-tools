// Function: sub_67F250
// Address: 0x67f250
//
FILE *sub_67F250()
{
  FILE *result; // rax

  sub_720F70(&qword_4CFDEB0);
  result = stderr;
  if ( qword_4F07510 != stderr )
  {
    sub_720F70(&qword_4F07510);
    result = stderr;
  }
  qword_4F07510 = result;
  return result;
}
