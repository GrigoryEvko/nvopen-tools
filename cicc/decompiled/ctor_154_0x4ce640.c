// Function: ctor_154
// Address: 0x4ce640
//
char *ctor_154()
{
  char *result; // rax

  result = getenv("LLVM_OVERRIDE_PRODUCER");
  if ( !result )
    result = a701;
  qword_4F9F888 = result;
  return result;
}
