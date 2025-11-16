// Function: sub_721870
// Address: 0x721870
//
void *__fastcall sub_721870(__int64 a1, size_t a2, __off_t a3)
{
  void *result; // rax

  if ( fseek(qword_4F078D8, a3 + a2, 0) )
    return 0;
  if ( fputc(0, qword_4F078D8) == -1 )
    return 0;
  if ( fflush(qword_4F078D8) )
    return 0;
  result = mmap(0, a2, 3, 2, fd, a3);
  if ( result == (void *)-1LL )
    return 0;
  return result;
}
