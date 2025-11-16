// Function: sub_7219E0
// Address: 0x7219e0
//
int sub_7219E0()
{
  int result; // eax

  if ( qword_4F078D8 )
    result = fclose(qword_4F078D8);
  qword_4F078D8 = 0;
  return result;
}
