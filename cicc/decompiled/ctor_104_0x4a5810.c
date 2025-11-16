// Function: ctor_104
// Address: 0x4a5810
//
int ctor_104()
{
  byte_4F92D70 = getenv("LIBNVVM_DISABLE_CONCURRENT_API") != 0;
  return __cxa_atexit(sub_12B9970, &byte_4F92D70, &qword_4A427C0);
}
